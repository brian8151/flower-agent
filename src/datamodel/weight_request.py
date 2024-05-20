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
    kernel_regularizer: Optional[str] = None
    bias_regularizer: Optional[str] = None
    activity_regularizer: Optional[str] = None
    kernel_constraint: Optional[str] = None
    bias_constraint: Optional[str] = None
    build_config: Optional[Dict[str, List[Optional[int]]]] = None


class Layer(BaseModel):
    module: str
    class_name: str
    config: LayerConfig
    registered_name: Optional[str] = None


class ModelConfig(BaseModel):
    name: str
    layers: List[Layer]


class Model(BaseModel):
    class_name: str
    config: ModelConfig
    keras_version: str
    backend: str


class WeightRequest(BaseModel):
    model: Model
