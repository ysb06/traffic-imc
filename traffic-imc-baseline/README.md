# traffic-imc-baseline

Unified training runner for Traffic-IMC baseline models.

## Supported Models
- dcrnn
- agcrn
- stgcn
- lstm
- mlcaformer

## Prerequisites
1. Python `3.11+`
2. `traffic-imc-dataset` installed (editable install recommended)
3. Dataset files prepared (either generated or downloaded)
4. Weights & Biases login configured

```bash
wandb login
```

## Install (From Repository Root)
Run from the root directory (`traffic-imc/`):

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -e ./traffic-imc-dataset -e ./traffic-imc-baseline
```

## Quick Start (From Repository Root)
```bash
traffic-imc --model dcrnn --config ./configs/baseline/dcrnn.yaml
```

Module form:

```bash
python -m traffic_imc_baseline --model dcrnn --config ./configs/baseline/dcrnn.yaml
```

## Outputs
- Checkpoints and trainer outputs: `output/<model>/<name_key>_<code>/`
- WandB local run artifacts: `wandb/`

WandB logging is mandatory in this runner.
