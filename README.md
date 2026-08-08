# TSB-AD with FuKAN

This repository is based on [TSB-AD](https://github.com/thedatumorg/TSB-AD) and
integrates FuKAN as a built-in detector.

## Installation

```bash
pip install -e .
```

## Quick start

```python
from TSB_AD.model_wrapper import run_Semisupervise_AD

score = run_Semisupervise_AD(
    "FuKAN",
    data_train, data_test,
    ...  # hyper-parameters can be set in TSB_AD/HP_list.py
)
```

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

