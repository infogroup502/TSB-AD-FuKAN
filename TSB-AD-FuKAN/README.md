# TSB-AD with FuKAN

A time-series anomaly detection benchmark integrated with the **FuKAN**
detector.

This repository is based on [TSB-AD](https://github.com/thedatumorg/TSB-AD)
and adds FuKAN as a built-in semi-supervised anomaly detector.  FuKAN learns
bidirectional local/global fuzzy-membership consistency through
Kolmogorov-Arnold networks (KAN).  The benchmark pipeline, evaluation
metrics and dataset handling come from TSB-AD and are kept unchanged.

## FuKAN

FuKAN maps each sliding context into fuzzy memberships and uses KAN layers to
enforce consistency between local and global views in both forward and
backward directions.  Anomaly scores are derived from the bidirectional
fuzzy-membership reconstruction error.

```bash
pip install -e .
```

Run FuKAN through the TSB-AD semantics wrapper:

```python
from TSB_AD.model_wrapper import run_Semisupervise_AD

score = run_Semisupervise_AD(
    "FuKAN",
    data_train, data_test,
    win_size=30,
    seq_len=2,
    local_size=(3,),
    global_size=(7,),
    fuzzy_sets=4,
    top_k=3,
    epochs=3,
    lr=5e-4,
    value_weight=1.0,
    normalize_mode="global",
    score_agg="max",
    scheduler_type="step",
    batch_size=128,
)
```

### Notes on the FuKAN configuration

- `epochs`: the model reaches saturation quickly under the step-decayed
  learning rate; 3 epochs are sufficient in practice.
- `normalize_mode="global"`: standardize with dataset statistics instead of a
  per-window z-score, which keeps absolute amplitude anomalies.
- `score_agg="max"`: aggregate window scores per timestamp over all covering
  windows rather than using a single trailing score.
- `value_weight` / `next_step_weight`: optional auxiliary reconstruction /
  prediction heads that can be toggled to zero to disable them.

## Directory layout

```
TSB_AD/
├── models/
│   ├── FuKAN.py          # FuKAN detector implementation
│   └── ...
├── model_wrapper.py      # detector adapters incl. run_FuKAN
├── HP_list.py            # hyper-parameter search grids and optimal configs
├── evaluation/           # metrics and evaluation helpers
└── utils/                # sliding-window and utilities
```

## License

See `LICENSE`.  The dataset and benchmark scripts are not included in this
repository; please fetch `Datasets/` and the official evaluation scripts from
the upstream TSB-AD repository.