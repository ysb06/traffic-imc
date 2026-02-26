# traffic-imc

## Overview
Traffic-IMC is based on the paper *Traffic-IMC: A Benchmark for Robustness against Real-World Sensor Failures in Complex Urban Road Networks*.  
The work focuses on interrupted traffic flow settings in urban networks and uses standard node-link topology as a key structural foundation.

## Subprojects
- `traffic-imc-dataset`  
  Dataset generation pipeline for METR-IMC raw data and interpolation-based subsets.
- `traffic-imc-baseline`  
  Baseline model training/evaluation pipeline driven by a unified command-line interface.

## Quick Start
Run from the root project directory:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ./traffic-imc-dataset -e ./traffic-imc-baseline
```

## Run Workflow
1. Generate dataset:

```bash
traffic-imc-dataset --api-key "YOUR_DATA_API_KEY"
```

Or use environment variable:

```bash
export DATA_API_KEY="YOUR_DATA_API_KEY"
traffic-imc-dataset
```

`--config-dir` is optional and defaults to `./configs`. If you want to specify a different directory for configuration files, you can use the `--config-dir` flag:

```bash
traffic-imc-dataset --api-key "YOUR_DATA_API_KEY" --config-dir ./configs
```

Or download the pre-generated dataset:

- Download link: [Google Drive (pre-generated METR-IMC dataset)](https://drive.google.com/drive/folders/1xgilXK2-ojll5PGKSm4-QfKtO335t04q?usp=sharing)
- Place the downloaded `metr-imc` folder at `./datasets/metr-imc` in the project root.

2. Run a baseline model:

```bash
traffic-imc --model agcrn --config traffic-imc-baseline/configs/agcrn.yaml
```
