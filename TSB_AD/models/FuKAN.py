"""FUkan detector for TSB-AD.

This module adapts the official implementation from
https://github.com/infogroup502/FuKAN to TSB-AD's detector API.  The
dataset-specific loading, thresholding and metric code from the original
``Solver`` are intentionally kept outside the detector: TSB-AD owns those
parts of the benchmark pipeline.
"""

import math
from typing import List, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from .base import BaseDetector


class _WindowDataset(Dataset):
    """Generate sliding windows lazily to avoid materializing large datasets.

    Each sample returns ``(window, next_value)`` where ``next_value`` is the
    value immediately following the window (used by the optional next-step
    prediction head).  For the last window that touches the end of the series
    the final value is replicated.
    """

    def __init__(self, data: np.ndarray, window_size: int):
        tensor = torch.as_tensor(data, dtype=torch.float32)
        if len(tensor) < window_size:
            padding = tensor[-1:].repeat(window_size - len(tensor), 1)
            tensor = torch.cat((tensor, padding), dim=0)
        self.data = tensor
        self.window_size = window_size
        self.sample_num = len(tensor) - window_size + 1
        next_vals = torch.empty(self.sample_num, tensor.shape[-1], dtype=tensor.dtype)
        for i in range(self.sample_num):
            end = i + window_size
            next_vals[i] = tensor[end] if end < len(tensor) else tensor[-1]
        self.next_values = next_vals

    def __len__(self) -> int:
        return self.sample_num

    def __getitem__(self, index: int):
        window = self.data[index:index + self.window_size]
        return window, self.next_values[index]


class KANLinear(nn.Module):
    """One Kolmogorov-Arnold layer used by the official FUkan model."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        grid_size: int = 3,
        spline_order: int = 1,
        scale_noise: float = 0.1,
        scale_base: float = 1.0,
        scale_spline: float = 1.0,
        enable_standalone_scale_spline: bool = True,
        grid_eps: float = 0.02,
        grid_range: Tuple[float, float] = (-1.0, 1.0),
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.grid_size = grid_size
        self.spline_order = spline_order
        self.scale_noise = scale_noise
        self.scale_base = scale_base
        self.scale_spline = scale_spline
        self.enable_standalone_scale_spline = enable_standalone_scale_spline
        self.grid_eps = grid_eps
        self.base_activation = nn.SiLU()

        grid_step = (grid_range[1] - grid_range[0]) / grid_size
        grid = (
            torch.arange(-spline_order, grid_size + spline_order + 1)
            * grid_step
            + grid_range[0]
        ).expand(in_features, -1).contiguous()
        self.register_buffer("grid", grid)

        self.base_weight = nn.Parameter(torch.empty(out_features, in_features))
        self.spline_weight = nn.Parameter(
            torch.empty(out_features, in_features, grid_size + spline_order)
        )
        if enable_standalone_scale_spline:
            self.spline_scaler = nn.Parameter(torch.empty(out_features, in_features))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.base_weight, a=math.sqrt(5) * self.scale_base)
        with torch.no_grad():
            noise = (
                torch.rand(self.grid_size + 1, self.in_features, self.out_features)
                - 0.5
            ) * self.scale_noise / self.grid_size
            scale = 1.0 if self.enable_standalone_scale_spline else self.scale_spline
            self.spline_weight.copy_(
                scale
                * self.curve2coeff(
                    self.grid.T[self.spline_order:-self.spline_order], noise
                )
            )
            if self.enable_standalone_scale_spline:
                nn.init.kaiming_uniform_(
                    self.spline_scaler, a=math.sqrt(5) * self.scale_spline
                )

    def b_splines(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 2 or x.size(1) != self.in_features:
            raise ValueError(
                f"KANLinear expected (*, {self.in_features}), got {tuple(x.shape)}"
            )
        grid = self.grid
        x = x.unsqueeze(-1)
        bases = ((x >= grid[:, :-1]) & (x < grid[:, 1:])).to(x.dtype)
        for order in range(1, self.spline_order + 1):
            bases = (
                (x - grid[:, :-(order + 1)])
                / (grid[:, order:-1] - grid[:, :-(order + 1)])
                * bases[:, :, :-1]
            ) + (
                (grid[:, order + 1:] - x)
                / (grid[:, order + 1:] - grid[:, 1:-order])
                * bases[:, :, 1:]
            )
        return bases.contiguous()

    def curve2coeff(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        a = self.b_splines(x).transpose(0, 1)
        b = y.transpose(0, 1)
        if hasattr(torch.linalg, "lstsq"):
            solution = torch.linalg.lstsq(a, b).solution
        else:  # PyTorch 1.8 compatibility.
            solution = torch.matmul(torch.pinverse(a), b)
        return solution.permute(2, 0, 1).contiguous()

    @property
    def scaled_spline_weight(self) -> torch.Tensor:
        if self.enable_standalone_scale_spline:
            return self.spline_weight * self.spline_scaler.unsqueeze(-1)
        return self.spline_weight

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.size(-1) != self.in_features:
            raise ValueError(
                f"KANLinear expected last dimension {self.in_features}, "
                f"got {x.size(-1)}"
            )
        original_shape = x.shape
        x = x.reshape(-1, self.in_features)
        base_output = F.linear(self.base_activation(x), self.base_weight)
        spline_output = F.linear(
            self.b_splines(x).reshape(x.size(0), -1),
            self.scaled_spline_weight.reshape(self.out_features, -1),
        )
        return (base_output + spline_output).reshape(
            *original_shape[:-1], self.out_features
        )


class FuzzyKAN(nn.Module):
    """Map input values to fuzzy memberships and infer output memberships."""

    def __init__(self, input_size: int, fuzzy_sets: int, output_size: int):
        super().__init__()
        self.sigma = nn.Parameter(torch.rand(input_size, fuzzy_sets))
        self.mu = nn.Parameter(torch.rand(input_size, fuzzy_sets))
        self.kan = KANLinear(input_size, output_size)
        # Secondary head that predicts the *raw* target values from the same
        # fuzzy memberships.  Its error gives a standard reconstruction-style
        # anomaly signal that is not suppressed when inputs fall outside the
        # learned fuzzy regions.
        self.value_head = KANLinear(input_size, output_size)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        membership = torch.exp(
            -((x.unsqueeze(-1) - self.mu) ** 2)
            / (2.0 * (self.sigma.square() + 1e-6))
        )
        # KAN learns across context positions independently for every fuzzy set.
        membership_transposed = membership.transpose(-1, -2)
        inferred = self.kan(membership_transposed).transpose(-1, -2)
        value_pred = self.value_head(membership_transposed).transpose(-1, -2)
        return inferred, membership, value_pred


class FuKANNetwork(nn.Module):
    """Four Fuzzy-KAN paths for forward/backward contrastive learning."""

    def __init__(
        self,
        local_size: Sequence[int],
        global_size: Sequence[int],
        seq_len: int = 2,
        fuzzy_sets: int = 4,
    ):
        super().__init__()
        if len(local_size) != len(global_size):
            raise ValueError("local_size and global_size must have equal lengths")
        self.local_size = tuple(int(size) for size in local_size)
        self.global_size = tuple(int(size) for size in global_size)
        self.seq_len = int(seq_len)

        local_dims = [size * self.seq_len for size in self.local_size]
        global_dims = [size * self.seq_len for size in self.global_size]
        self.local_to_global_front = nn.ModuleList(
            FuzzyKAN(local, fuzzy_sets, global_)
            for local, global_ in zip(local_dims, global_dims)
        )
        self.global_to_local_front = nn.ModuleList(
            FuzzyKAN(global_, fuzzy_sets, local)
            for local, global_ in zip(local_dims, global_dims)
        )
        self.local_to_global_back = nn.ModuleList(
            FuzzyKAN(local, fuzzy_sets, global_)
            for local, global_ in zip(local_dims, global_dims)
        )
        self.global_to_local_back = nn.ModuleList(
            FuzzyKAN(global_, fuzzy_sets, local)
            for local, global_ in zip(local_dims, global_dims)
        )
        # B4: auxiliary next-step prediction head.  It consumes the local
        # context of every window position and forecasts the value one step
        # ahead, complementing the contrastive path on low-amplitude /
        # long-segment anomalies.
        self.next_step_head = NextStepKAN(local_dims[0], fuzzy_sets)

    def forward(
        self,
        local_front: List[torch.Tensor],
        global_front: List[torch.Tensor],
        local_back: List[torch.Tensor],
        global_back: List[torch.Tensor],
    ):
        results = []
        for index in range(len(self.local_size)):
            (
                pred_global_f,
                membership_local_f,
                value_global_f,
            ) = self.local_to_global_front[index](local_front[index])
            (
                pred_local_f,
                membership_global_f,
                value_local_f,
            ) = self.global_to_local_front[index](global_front[index])
            (
                pred_global_b,
                membership_local_b,
                value_global_b,
            ) = self.local_to_global_back[index](local_back[index])
            (
                pred_local_b,
                membership_global_b,
                value_local_b,
            ) = self.global_to_local_back[index](global_back[index])
            results.append(
                (
                    pred_global_f,
                    membership_global_f,
                    pred_local_f,
                    membership_local_f,
                    pred_global_b,
                    membership_global_b,
                    pred_local_b,
                    membership_local_b,
                    value_global_f,
                    value_local_f,
                    value_global_b,
                    value_local_b,
                )
            )
        # next_pred: (B, L, M) — the forecast of the next timestamp for every
        # window position, derived from that position's local context.
        next_pred = self.next_step_head(local_front[0])
        return results, next_pred


class NextStepKAN(nn.Module):
    """Predict the value one step ahead from the local context of each position.

    Consumes ``local_front`` shaped ``(B, L, M, ctx)`` (the same context that
    feeds the contrastive local→global path) and produces a per-position,
    per-feature forecast of the next timestamp: ``(B, L, M)``.  Its error is a
    standard prediction-style anomaly signal that complements the contrastive
    local/global path - especially on low-amplitude / long-segment anomalies
    where the contrastive signal collapses.
    """

    def __init__(self, input_size: int, fuzzy_sets: int):
        super().__init__()
        # Membership centres/widths are indexed by context position and shared
        # across features, mirroring the main FuzzyKAN module.
        self.sigma = nn.Parameter(torch.rand(input_size, fuzzy_sets))
        self.mu = nn.Parameter(torch.rand(input_size, fuzzy_sets))
        self.kan = KANLinear(input_size, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, L, M, ctx)  (last axis is the local context steps)
        membership = torch.exp(
            -((x.unsqueeze(-1) - self.mu) ** 2)
            / (2.0 * (self.sigma.square() + 1e-6))
        )
        # membership: (B, L, M, ctx, fuzzy_sets)
        membership_transposed = membership.transpose(-1, -2)
        # membership_transposed: (B, L, M, fuzzy_sets, ctx)
        flat = membership_transposed.reshape(
            -1, membership_transposed.size(-1)
        )
        out = self.kan(flat)  # (B*L*M*fuzzy_sets, 1)
        out = out.reshape(*membership_transposed.shape[:-1])  # (B,L,M,fuzzy_sets)
        # Average the fuzzy-set heads and return one scalar per feature.
        return out.mean(dim=-1)  # (B, L, M)


def _gaussian_smooth(scores: np.ndarray, sigma: float) -> np.ndarray:
    """Convolution-based Gaussian smoothing, segment/VUS friendly."""
    radius = max(1, int(3 * sigma))
    x = np.arange(-radius, radius + 1, dtype=np.float64)
    kernel = np.exp(-0.5 * (x / sigma) ** 2)
    kernel /= kernel.sum()
    return np.convolve(scores, kernel, mode="same")


class FuKAN(BaseDetector):
    """Fuzzy KAN-based bidirectional time-series anomaly detector.

    Parameters follow the official implementation. ``fit`` learns only from
    the normal training prefix supplied by TSB-AD, while
    ``decision_function`` returns one continuous anomaly score per timestamp.
    """

    def __init__(
        self,
        win_size: int = 20,
        seq_len: int = 2,
        local_size: Sequence[int] = (1,),
        global_size: Sequence[int] = (10,),
        fuzzy_sets: int = 4,
        top_k: int = 3,
        batch_size: int = 128,
        epochs: int = 2,
        lr: float = 1e-4,
        device: str = "auto",
        contamination: float = 0.1,
        value_weight: float = 1.0,
        normalize_mode: str = "window",
        score_agg: str = "tail",
        score_smooth: float = 0.0,
        scheduler_type: str = "step",
        next_step_weight: float = 0.0,
    ):
        super().__init__(contamination=contamination)
        self.win_size = int(win_size)
        self.seq_len = int(seq_len)
        self.local_size = tuple(int(size) for size in local_size)
        self.global_size = tuple(int(size) for size in global_size)
        self.fuzzy_sets = int(fuzzy_sets)
        self.top_k = int(top_k)
        self.batch_size = int(batch_size)
        self.epochs = int(epochs)
        self.lr = float(lr)
        self.device = device
        self.value_weight = float(value_weight)
        self.normalize_mode = str(normalize_mode)
        self.score_agg = str(score_agg)
        self.score_smooth = float(score_smooth)
        self.scheduler_type = str(scheduler_type)
        self.next_step_weight = float(next_step_weight)

        self._validate_hyperparameters()
        self._device = self._resolve_device(device)
        self.model = FuKANNetwork(
            local_size=self.local_size,
            global_size=self.global_size,
            seq_len=self.seq_len,
            fuzzy_sets=self.fuzzy_sets,
        ).to(self._device)
        self._mean = None
        self._std = None
        self.__anomaly_score = None
        self._next_values_available = False
        self._batch_next_values = None
        self._next_values_available = False  # kept for API compatibility

    def _validate_hyperparameters(self) -> None:
        if self.win_size < 2:
            raise ValueError("win_size must be at least 2")
        if self.seq_len < 1:
            raise ValueError("seq_len must be positive")
        if not self.local_size or len(self.local_size) != len(self.global_size):
            raise ValueError("local_size and global_size must be non-empty and aligned")
        if any(size < 1 for size in self.local_size + self.global_size):
            raise ValueError("all local/global context sizes must be positive")
        if max(self.local_size) * self.seq_len > self.win_size:
            raise ValueError("local_size * seq_len cannot exceed win_size")
        if min(self.fuzzy_sets, self.top_k, self.batch_size, self.epochs) < 1:
            raise ValueError("fuzzy_sets, top_k, batch_size and epochs must be positive")
        if self.value_weight < 0:
            raise ValueError("value_weight must be non-negative")
        if self.normalize_mode not in ("window", "global"):
            raise ValueError("normalize_mode must be 'window' or 'global'")
        if self.score_agg not in ("tail", "max"):
            raise ValueError("score_agg must be 'tail' or 'max'")
        if self.score_smooth < 0:
            raise ValueError("score_smooth must be non-negative")
        if self.scheduler_type not in ("step", "cosine"):
            raise ValueError("scheduler_type must be 'step' or 'cosine'")
        if self.next_step_weight < 0:
            raise ValueError("next_step_weight must be non-negative")

    @staticmethod
    def _resolve_device(device: str) -> torch.device:
        if device == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        resolved = torch.device(device)
        if resolved.type == "cuda" and not torch.cuda.is_available():
            raise ValueError("CUDA was requested but is not available")
        return resolved

    @staticmethod
    def _as_2d_array(data: np.ndarray, name: str) -> np.ndarray:
        array = np.asarray(data, dtype=np.float32)
        if array.ndim == 1:
            array = array.reshape(-1, 1)
        if array.ndim != 2 or len(array) == 0:
            raise ValueError(f"{name} must have shape (n_samples, n_features)")
        if not np.isfinite(array).all():
            raise ValueError(f"{name} contains NaN or infinite values")
        return array

    def _standardize(self, data: np.ndarray) -> np.ndarray:
        return (data - self._mean) / self._std

    def _window_normalize(self, x: torch.Tensor) -> torch.Tensor:
        if self.normalize_mode == "global":
            # Data entering the windowing step is already standardized with
            # training-set statistics (``_standardize``).  A per-window
            # z-score would compress amplitude spikes into the same 1-3
            # sigma band as flat-region noise, so pass through unchanged to
            # preserve the absolute-magnitude anomaly signal.
            return x
        mean = x.mean(dim=1, keepdim=True)
        variance = x.var(dim=1, keepdim=True, unbiased=False)
        return (x - mean) / torch.sqrt(variance + 1e-5)

    def _make_contexts(
        self, x: torch.Tensor
    ) -> Tuple[List[torch.Tensor], ...]:
        """Reproduce the official historical sampling and its reverse order.

        The global context starts one ``local_count`` step behind the current
        position (instead of at the current position itself).  This avoids the
        self-comparison leak where the point being scored appears in both the
        local and the global view, which systematically depressed anomaly
        scores for isolated spikes.
        """
        _, length, _ = x.shape
        time = torch.arange(length, device=x.device)
        local_front, global_front, local_back, global_back = [], [], [], []

        for local_size, global_size in zip(self.local_size, self.global_size):
            local_count = self.seq_len * local_size
            local_offsets = torch.arange(
                -local_count + 1, 1, device=x.device
            )
            local_indices = (time[:, None] + local_offsets).clamp(0, length - 1)
            local = x[:, local_indices, :].permute(0, 1, 3, 2).contiguous()

            groups = torch.arange(global_size, device=x.device) * local_count
            within_group = torch.arange(self.seq_len, device=x.device)
            global_offsets = -(
                local_count + groups[:, None] + within_group[None, :]
            ).reshape(-1)
            global_indices = (time[:, None] + global_offsets).clamp(0, length - 1)
            global_ = x[:, global_indices, :].permute(0, 1, 3, 2).contiguous()

            local_front.append(local)
            global_front.append(global_)
            local_back.append(local.flip(-1))
            global_back.append(global_.flip(-1))

        return local_front, global_front, local_back, global_back

    def _network_outputs(self, windows: torch.Tensor):
        """Return (contexts, (outputs, next_pred)) for a batch of windows."""
        normalized = self._window_normalize(windows)
        contexts = self._make_contexts(normalized)
        outputs, next_pred = self.model(*contexts)
        return contexts, (outputs, next_pred)

    def _contrastive_loss(self, windows: torch.Tensor) -> torch.Tensor:
        loss = windows.new_zeros(())
        contexts, (outputs, next_pred) = self._network_outputs(windows)
        local_front, global_front, local_back, global_back = contexts
        for index, output in enumerate(outputs):
            for prediction, target in (
                (output[0], output[1]),
                (output[2], output[3]),
                (output[4], output[5]),
                (output[6], output[7]),
            ):
                # Normalize by the fuzzy-set dimension so membership errors
                # and value errors share a comparable per-slot scale.
                loss = loss + F.mse_loss(prediction, target) / self.fuzzy_sets
            if self.value_weight > 0:
                for values, prediction in (
                    (global_front[index], output[8]),
                    (local_front[index], output[9]),
                    (global_back[index], output[10]),
                    (local_back[index], output[11]),
                ):
                    loss = loss + self.value_weight * F.mse_loss(
                        prediction, values.unsqueeze(-1).expand_as(prediction)
                    ) / self.fuzzy_sets
        if self.next_step_weight > 0:
            # Auxiliary next-step prediction loss (B4).
            target = self._next_step_target(windows)
            loss = loss + self.next_step_weight * F.mse_loss(next_pred, target)
        return loss

    def _next_step_target(self, windows: torch.Tensor) -> torch.Tensor:
        """Per-position next-step target inside a batch of windows.

        The target of position ``i`` is the value at ``i+1`` within the
        window; the last position ``L-1`` copies its own value (there is no
        in-window successor).  Using the same target in training and scoring
        keeps the signal consistent and causal - the model only uses
        information available up to the position being predicted.
        """
        return torch.cat((windows[:, 1:, :], windows[:, -1:, :]), dim=1)

    def _next_step_score(self, windows: torch.Tensor,
                         next_pred: torch.Tensor) -> torch.Tensor:
        """Per-position, per-feature next-step prediction error."""
        target = self._next_step_target(windows)
        return (next_pred - target).square()  # (B, L, M)

    def _window_scores(self, windows: torch.Tensor) -> torch.Tensor:
        forward_difference = windows.new_zeros(windows.shape)
        backward_difference = windows.new_zeros(windows.shape)

        contexts, (outputs, next_pred) = self._network_outputs(windows)
        local_front, global_front, local_back, global_back = contexts
        for index, output in enumerate(outputs):
            forward_difference += (
                (output[0] - output[1]).square().sum(dim=(-1, -2)) / self.fuzzy_sets
                + (output[2] - output[3]).square().sum(dim=(-1, -2)) / self.fuzzy_sets
            )
            backward_difference += (
                (output[4] - output[5]).square().sum(dim=(-1, -2)) / self.fuzzy_sets
                + (output[6] - output[7]).square().sum(dim=(-1, -2)) / self.fuzzy_sets
            )
        if self.next_step_weight > 0:
            forward_difference = forward_difference + self.next_step_weight * (
                self._next_step_score(windows, next_pred)
            )
            backward_difference = backward_difference + self.next_step_weight * (
                self._next_step_score(windows, next_pred)
            )
        for index, output in enumerate(outputs):
            if self.value_weight > 0:
                forward_difference += self.value_weight * (
                    (output[8] - global_front[index].unsqueeze(-1).expand_as(output[8]))
                    .square()
                    .sum(dim=(-1, -2))
                    / self.fuzzy_sets
                    + (output[9] - local_front[index].unsqueeze(-1).expand_as(output[9]))
                    .square()
                    .sum(dim=(-1, -2))
                    / self.fuzzy_sets
                )
                backward_difference += self.value_weight * (
                    (output[10] - global_back[index].unsqueeze(-1).expand_as(output[10]))
                    .square()
                    .sum(dim=(-1, -2))
                    / self.fuzzy_sets
                    + (output[11] - local_back[index].unsqueeze(-1).expand_as(output[11]))
                    .square()
                    .sum(dim=(-1, -2))
                    / self.fuzzy_sets
                )

        k = min(self.top_k, windows.shape[-1])
        forward_score = forward_difference.topk(k, dim=-1).values.mean(dim=-1)
        backward_score = backward_difference.topk(k, dim=-1).values.mean(dim=-1)
        return torch.maximum(forward_score, backward_score)

    def fit(self, X: np.ndarray, y=None):
        X = self._as_2d_array(X, "X")
        self._set_n_classes(y)
        self._mean = X.mean(axis=0, keepdims=True)
        self._std = X.std(axis=0, keepdims=True)
        self._std = np.where(self._std == 0, 1e-8, self._std)
        train = self._standardize(X)

        loader = DataLoader(
            _WindowDataset(train, self.win_size),
            batch_size=self.batch_size,
            shuffle=True,
            drop_last=False,
        )
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        if self.scheduler_type == "cosine":
            # Smooth decay over the whole training budget.  The legacy StepLR
            # halves lr every epoch and vanishes after ~2 epochs, leaving most
            # of a 30-epoch run nearly static.
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=max(1, self.epochs), eta_min=1e-5
            )
        else:
            scheduler = torch.optim.lr_scheduler.StepLR(
                optimizer, step_size=1, gamma=0.5
            )

        self.model.train()
        for _ in range(self.epochs):
            for windows, next_vals in loader:
                windows = windows.to(self._device)
                next_vals = next_vals.to(self._device)
                # Expose next values so the auxiliary next-step head can
                # supervise the value that follows the window.
                self._next_values_available = True
                self._batch_next_values = next_vals
                optimizer.zero_grad()
                loss = self._contrastive_loss(windows)
                loss.backward()
                optimizer.step()
            scheduler.step()
            # Release cached intermediate tensors between epochs to lower
            # peak GPU memory (safe: does not change scores or the model).
            if self._device.type == "cuda":
                torch.cuda.empty_cache()
        self._next_values_available = False
        if self._device.type == "cuda":
            torch.cuda.empty_cache()

        self.decision_scores_ = self._score_array(X)
        self._process_decision_scores()
        return self

    def _score_array(self, X: np.ndarray) -> np.ndarray:
        if self._mean is None or self._std is None:
            raise RuntimeError("FuKAN must be fitted before scoring")
        if X.shape[1] != self._mean.shape[1]:
            raise ValueError(
                f"X has {X.shape[1]} features, expected {self._mean.shape[1]}"
            )

        data = self._standardize(X)
        original_length = len(data)
        window_size = self.win_size

        # Score every timestamp with stride-1 sliding windows instead of one
        # scalar per non-overlapping block.  Each window yields a score for
        # every position inside it; the score of the *last* position is
        # assigned to the timestamp the window ends at, so every timestamp
        # receives its own score (the first window_size-1 timestamps reuse the
        # first window's trailing score, matching the alignment used by other
        # TSB-AD neural detectors).
        if original_length < window_size:
            data = np.concatenate(
                (data, np.repeat(data[-1:], window_size - original_length, axis=0)),
                axis=0,
            )

        windows = np.lib.stride_tricks.sliding_window_view(
            data, window_size, axis=0
        )
        # sliding_window_view appends the window axis at the end, giving
        # (n - window_size + 1, n_features, window_size); move it to the
        # expected (batch, window, features) layout.
        windows = windows.transpose(0, 2, 1)

        # Per-position scores for every window: ``all_window_scores[i, j]``
        # is the score of timestamp ``i + j`` as seen from the window
        # starting at ``i``.
        all_window_scores = np.empty(
            (len(windows), window_size), dtype=np.float64
        )
        self.model.eval()
        with torch.no_grad():
            for start in range(0, len(windows), self.batch_size):
                batch = np.ascontiguousarray(windows[start:start + self.batch_size])
                batch_tensor = torch.as_tensor(batch, dtype=torch.float32).to(
                    self._device
                )
                scores = self._window_scores(batch_tensor).cpu().numpy()
                all_window_scores[start:start + len(batch)] = scores

        if self.score_agg == "tail":
            # Legacy behavior: keep only the last position of each window
            # and reuse the first window's trailing score for the initial
            # window_size-1 timestamps (constant head padding).
            trailing = all_window_scores[:, -1]
            result = np.empty(len(windows) + window_size - 1, dtype=np.float64)
            result[:window_size - 1] = trailing[0]
            result[window_size - 1:] = trailing
        else:  # "max"
            # Score every timestamp with the maximum over all windows that
            # cover it.  Removes the constant head pad and keeps anomalies
            # inside long segments visible (any window that contains the
            # anomalous structure contributes a high score).
            result = np.full(
                len(windows) + window_size - 1, -np.inf, dtype=np.float64
            )
            covered = (
                np.arange(len(windows))[:, None]
                + np.arange(window_size)[None, :]
            )
            np.maximum.at(result, covered, all_window_scores)
        return result[:original_length].astype(np.float64, copy=False)

    def decision_function(self, X: np.ndarray) -> np.ndarray:
        X = self._as_2d_array(X, "X")
        self.__anomaly_score = self._score_array(X)
        if self.score_smooth > 0:
            self.__anomaly_score = _gaussian_smooth(
                self.__anomaly_score, self.score_smooth * self.win_size
            )
        return self.__anomaly_score

    def anomaly_score(self) -> np.ndarray:
        return self.__anomaly_score
