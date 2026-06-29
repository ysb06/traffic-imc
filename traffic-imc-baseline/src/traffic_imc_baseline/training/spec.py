from dataclasses import dataclass
from typing import Any, Callable, Generic, TypeVar

from traffic_imc_dataset.utils import PathConfig

from .config import StrictBaseModel


class StrictParams(StrictBaseModel):
    pass


DataParamsT = TypeVar("DataParamsT", bound=StrictParams)
ModelParamsT = TypeVar("ModelParamsT", bound=StrictParams)


@dataclass(frozen=True)
class ValidatedParams(Generic[DataParamsT, ModelParamsT]):
    data: DataParamsT
    model: ModelParamsT


@dataclass(frozen=True)
class TrainingSpec(Generic[DataParamsT, ModelParamsT]):
    key: str
    display_name: str
    output_subdir: str
    default_checkpoint_filename: str
    data_params_type: type[DataParamsT]
    model_params_type: type[ModelParamsT]
    build_datamodule: Callable[[DataParamsT, PathConfig], Any]
    build_model: Callable[[ModelParamsT, DataParamsT, Any, PathConfig], Any]

    def validate_params(
        self,
        data_params: dict[str, Any],
        model_params: dict[str, Any],
    ) -> ValidatedParams[DataParamsT, ModelParamsT]:
        return ValidatedParams(
            data=self.data_params_type.model_validate(data_params),
            model=self.model_params_type.model_validate(model_params),
        )
