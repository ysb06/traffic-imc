# traffic-imc-baseline

Unified training and evaluation runner for Traffic-IMC forecasting baselines.

## Supported Models

- `agcrn`
- `bigst`
- `dcrnn`
- `gwnet`
- `lstm`
- `mlcaformer`
- `mtgnn`
- `stgcn`
- `stid`

The model is selected by the YAML file passed to `--config`; the CLI does not take a separate `--model` argument.

## Prerequisites

1. Python `3.11+`
2. Traffic-IMC dataset files prepared
3. Weights & Biases login configured when using the default logger

## Install

From this subproject directory:

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -e .
```

If `traffic-imc-dataset` is checked out next to this subproject, install it before running training:

```bash
pip install -e ../traffic-imc-dataset
```

## Train a Baseline

Run DCRNN:

```bash
traffic-imc-baseline --config ./configs/baseline/dcrnn.yaml
```

Run another model by changing the config path:

```bash
traffic-imc-baseline --config ./configs/baseline/agcrn.yaml
traffic-imc-baseline --config ./configs/baseline/bigst.yaml
traffic-imc-baseline --config ./configs/baseline/gwnet.yaml
traffic-imc-baseline --config ./configs/baseline/lstm.yaml
traffic-imc-baseline --config ./configs/baseline/mlcaformer.yaml
traffic-imc-baseline --config ./configs/baseline/mtgnn.yaml
traffic-imc-baseline --config ./configs/baseline/stgcn.yaml
traffic-imc-baseline --config ./configs/baseline/stid.yaml
```
