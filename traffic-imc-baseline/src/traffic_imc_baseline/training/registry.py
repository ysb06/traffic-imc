from importlib import import_module

from .common import ModelAdapter

_ADAPTER_MODULES = {
    "agcrn": "traffic_imc_baseline.training.agcrn_training",
    "dcrnn": "traffic_imc_baseline.training.dcrnn_training",
    "lstm": "traffic_imc_baseline.training.lstm_training",
    "mlcaformer": "traffic_imc_baseline.training.mlcaformer_training",
    "stgcn": "traffic_imc_baseline.training.stgcn_training",
}


def list_models() -> list[str]:
    return sorted(_ADAPTER_MODULES.keys())


def get_adapter(model_key: str) -> ModelAdapter:
    normalized_key = model_key.lower()
    if normalized_key not in _ADAPTER_MODULES:
        supported = ", ".join(list_models())
        raise ValueError(f"Unsupported model '{model_key}'. Supported models: {supported}")

    module = import_module(_ADAPTER_MODULES[normalized_key])
    get_adapter_fn = getattr(module, "get_adapter")
    return get_adapter_fn()
