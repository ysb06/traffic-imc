# traffic-imc-baseline

Unified training runner for Traffic-IMC baseline models.

## Supported models
- dcrnn
- agcrn
- stgcn
- lstm
- mlcaformer

## Prerequisites
1. Use Python `3.11+`.
2. Ensure `traffic-imc-dataset` is available (editable install recommended).
3. Configure Weights & Biases login before training:

```bash
wandb login
```

## Quick start
From this project root:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ../traffic-imc-dataset -e .
wandb login
traffic-imc --model dcrnn --config configs/dcrnn.yaml
```

You can still run the module form if needed:

```bash
python -m traffic_imc_baseline --model dcrnn --config configs/dcrnn.yaml
```

## Configuration
Each model keeps its own YAML schema under `configs/`.

Common run section:

```yaml
run:
  name_key: "mice"
  code: 0
  seed: null
```

## Outputs
- Checkpoints and trainer outputs: `output/<model>/<name_key>_<code>/`
- WandB local run artifacts: `wandb/`

WandB logging is mandatory in this v1 runner.
