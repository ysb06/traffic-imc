# Traffic-IMC

Traffic-IMC is an urban road-network traffic-volume forecasting benchmark for evaluating imputation-aware forecasting pipelines under operational missingness. The benchmark is based on hourly traffic-volume records from Incheon, South Korea and a directed, reachability-aware road graph derived from Korean Standard Node-Link data.

This repository contains two subprojects:

- `traffic-imc-dataset`: dataset collection, quality control, graph construction, and imputed subset generation.
- `traffic-imc-baseline`: unified training and evaluation runner for forecasting baselines.

## Install and Run

From the repository root:

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -e ./traffic-imc-dataset
pip install -e ./traffic-imc-baseline
```

Run each command from its respective subproject directory. For example, to generate the dataset:

```bash
cd traffic-imc-dataset
traffic-imc-dataset --api-key "YOUR_DATA_API_KEY"
```

To run a baseline experiment:

```bash
cd traffic-imc-baseline
traffic-imc-baseline --config ./configs/baseline/agcrn.yaml
```
