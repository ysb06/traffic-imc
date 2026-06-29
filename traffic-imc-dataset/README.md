# traffic-imc-dataset

Dataset generation package for Traffic-IMC, an imputation-aware urban traffic-volume forecasting benchmark built from Incheon traffic-volume records and Korean Standard Node-Link road-network data.

## Prerequisites

- Python `3.11+`
- `data.go.kr` account
- Approved usage request and API key for the Incheon road traffic statistics OpenAPI

## Install

From this subproject directory:

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -e .
```

## data.go.kr API Key

1. Sign up and log in to `data.go.kr`.
2. Search for the Incheon traffic statistics OpenAPI: [Incheon Metropolitan City Road Traffic Statistics](https://www.data.go.kr/en/data/15113145/openapi.do).
3. Submit a usage request for the API.
4. After approval, use the issued general key (Decoding).

## Generate Dataset

Run commands from this subproject directory unless you intentionally use absolute paths. The default config directory is `./configs`, and relative dataset paths in the config files are resolved from the current working directory.

Pass the API key directly:

```bash
traffic-imc-dataset --api-key "YOUR_DATA_API_KEY"
```

Or use the environment variable:

```bash
export DATA_API_KEY="YOUR_DATA_API_KEY"
traffic-imc-dataset
```

To override the config directory:

```bash
traffic-imc-dataset --api-key "YOUR_DATA_API_KEY" --config-dir ./configs
```

## Use Pre-generated Dataset

If you do not need to run API collection and preprocessing, download the pre-generated dataset:

- Download link: [Google Drive (pre-generated Traffic-IMC dataset)](https://drive.google.com/drive/folders/1SuR0E9pW5pK0yCu_NpXyCRDaLr8KzZ4O?usp=sharing)
- Place the downloaded `traffic-imc` folder at `./datasets/traffic-imc` under this subproject directory.

The default baseline path config expects this location.
