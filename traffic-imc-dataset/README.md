# traffic-imc-dataset

## Overview
This project collects and processes Incheon metropolitan traffic volume data to build the METR-IMC dataset.

Generated outputs:
- 1 raw dataset
- 1 base subset
- 5 interpolated subsets (`mice`, `knn`, `bgcp`, `trmf`, `brits`)

## Prerequisites
- Python `3.11+`
- `data.go.kr` account
- Approved usage request and API key for the Incheon traffic OpenAPI

## Create Virtual Environment and Install
Run from the project root:

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -e .
```

## data.go.kr API Key Issuance
1. Sign up and log in to `data.go.kr`.
2. Search for the Incheon traffic statistics OpenAPI (https://www.data.go.kr/en/data/15113145/openapi.do).
3. Submit a usage request for the target API.
4. After approval, check your issued API key.
5. The general key (Decoding) is recommended.

## Run Dataset Generation

Inject API key directly at runtime (Recommended)

```bash
traffic-imc-dataset --api-key "YOUR_DATA_API_KEY" --config-dir ./configs
```

```bash
python -m traffic_imc_dataset --api-key "YOUR_DATA_API_KEY" --config-dir ./configs
```

Or use environment variable

```bash
export DATA_API_KEY="YOUR_DATA_API_KEY"
traffic-imc-dataset --config-dir ./configs
```

```bash
export DATA_API_KEY="YOUR_DATA_API_KEY"
python -m traffic_imc_dataset --config-dir ./configs
```

`--config-dir` is optional and defaults to `./configs`.

## Use Pre-generated Dataset

If you want to skip API collection and preprocessing, you can download the pre-generated dataset:

- Download link: [Google Drive (pre-generated METR-IMC dataset)](https://drive.google.com/drive/folders/1xgilXK2-ojll5PGKSm4-QfKtO335t04q?usp=sharing)
- Place the downloaded `metr-imc` folder at `./datasets/metr-imc` in the project root.
