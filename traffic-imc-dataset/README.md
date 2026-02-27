# traffic-imc-dataset

## Overview
This project collects and processes Incheon metropolitan traffic volume data to build the Traffic-IMC dataset.

Generated outputs:
- 1 raw dataset
- 1 base subset
- 5 interpolated subsets (`mice`, `knn`, `bgcp`, `trmf`, `brits`)

## Prerequisites
- Python `3.11+`
- `data.go.kr` account
- Approved usage request and API key for the Incheon traffic OpenAPI

## Install (From Repository Root)
Run from the root directory (`traffic-imc/`):

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -e ./traffic-imc-dataset
```

## data.go.kr API Key Issuance
1. Sign up and log in to `data.go.kr`.
2. Search for the Incheon traffic statistics OpenAPI ([link](https://www.data.go.kr/en/data/15113145/openapi.do)).
3. Submit a usage request for the target API.
4. After approval, check your issued API key.
5. The general key (Decoding) is recommended.

## Config Directory
Default config directory is `./configs/dataset`.

Required files:
- `config.yaml`
- `config_base.yaml`
- `config_mice.yaml`
- `config_knn.yaml`
- `config_bgcp.yaml`
- `config_trmf.yaml`
- `config_brits.yaml`

## Run Dataset Generation (From Repository Root)
Inject API key directly at runtime:

```bash
traffic-imc-dataset --api-key "YOUR_DATA_API_KEY"
```

```bash
python -m traffic_imc_dataset --api-key "YOUR_DATA_API_KEY"
```

Or use environment variable:

```bash
export DATA_API_KEY="YOUR_DATA_API_KEY"
traffic-imc-dataset
```

```bash
export DATA_API_KEY="YOUR_DATA_API_KEY"
python -m traffic_imc_dataset
```

To override config directory:

```bash
traffic-imc-dataset --api-key "YOUR_DATA_API_KEY" --config-dir ./configs/dataset
```

## Use Pre-generated Dataset
If you want to skip API collection and preprocessing, download the pre-generated dataset:

- Download link: [Google Drive (pre-generated Traffic-IMC dataset)](https://drive.google.com/drive/folders/1xgilXK2-ojll5PGKSm4-QfKtO335t04q?usp=sharing)
- Place the downloaded `traffic-imc` folder at `./datasets/traffic-imc` in the repository root.
