from typing import List, Optional

from pydantic import BaseModel, Field


class InitializerConfig(BaseModel):
    module: str
    class_name: str
    config: dict
    registered_name: Optional[str] = None


class LayerConfig(BaseModel):
    batch_input_shape: Optional[List[Optional[int]]] = None
    dtype: Optional[str] = None
    sparse: Optional[bool] = None
    ragged: Optional[bool] = None
    name: Optional[str] = None
    trainable: Optional[bool] = None
    units: Optional[int] = None
    activation: Optional[str] = None
    use_bias: Optional[bool] = None
    kernel_initializer: Optional[InitializerConfig] = None
    bias_initializer: Optional[InitializerConfig] = None
    kernel_regularizer: Optional[dict] = None
    bias_regularizer: Optional[dict] = None
    activity_regularizer: Optional[dict] = None
    kernel_constraint: Optional[dict] = None
    bias_constraint: Optional[dict] = None
    build_config: Optional[dict] = None


class Layer(BaseModel):
    module: str
    class_name: str
    config: LayerConfig
    registered_name: Optional[str] = None


class ModelConfig(BaseModel):
    name: str
    layers: List[Layer]


class FullModelConfig(BaseModel):
    class_name: str
    config: ModelConfig
    keras_version: str
    backend: str


class Model(BaseModel):
    class_name: str
    config: ModelConfig
    keras_version: str
    backend: str


class WeightRequest(BaseModel):
    model: Model


class DataItem(BaseModel):
    features: List[float] = Field(..., alias="features", description="prediction data features")


class PredictionRequest(BaseModel):
    domain_type: str = Field(..., alias="domainType", description="data seed domain type")
    workflow_trace_id: str = Field(..., alias="workflowTraceId", description="workflow trace id")
    data: List[DataItem] = Field(..., alias="data", description="prediction data list")


def convert_to_dict(obj):
    if isinstance(obj, BaseModel):
        obj_dict = obj.dict()
        for key, value in obj_dict.items():
            obj_dict[key] = convert_to_dict(value)
        return obj_dict
    elif isinstance(obj, list):
        return [convert_to_dict(item) for item in obj]
    else:
        return obj
