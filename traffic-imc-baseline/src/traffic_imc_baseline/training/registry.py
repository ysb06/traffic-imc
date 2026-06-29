from importlib import import_module

from .spec import TrainingSpec

_SPEC_MODULES = {
    "agcrn": "traffic_imc_baseline.training.specs.agcrn",
    "bigst": "traffic_imc_baseline.training.specs.bigst",
    "dcrnn": "traffic_imc_baseline.training.specs.dcrnn",
    "gwnet": "traffic_imc_baseline.training.specs.gwnet",
    "lstm": "traffic_imc_baseline.training.specs.lstm",
    "mlcaformer": "traffic_imc_baseline.training.specs.mlcaformer",
    "mtgnn": "traffic_imc_baseline.training.specs.mtgnn",
    "stgcn": "traffic_imc_baseline.training.specs.stgcn",
    "stid": "traffic_imc_baseline.training.specs.stid",
}


def list_models() -> list[str]:
    return sorted(_SPEC_MODULES.keys())


def get_spec(model_key: str) -> TrainingSpec:
    normalized_key = model_key.lower()
    if normalized_key not in _SPEC_MODULES:
        supported = ", ".join(list_models())
        raise ValueError(f"Unsupported model '{model_key}'. Supported models: {supported}")

    module = import_module(_SPEC_MODULES[normalized_key])
    get_spec_fn = getattr(module, "get_spec")
    return get_spec_fn()
