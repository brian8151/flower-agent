from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any


class InitializerConfig(BaseModel):
    module: str
    class_name: str
    config: Dict[str, Any]
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
    kernel_regularizer: Optional[Any] = None
    bias_regularizer: Optional[Any] = None
    activity_regularizer: Optional[Any] = None
    kernel_constraint: Optional[Any] = None
    bias_constraint: Optional[Any] = None
    build_config: Optional[Dict[str, Any]] = None


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



class WeightRequest(BaseModel):
    # model: FullModelConfig = Field(..., alias="model", description="Model configuration")
    model: FullModelConfig = Field(..., alias="model", description="Model configuration")


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
