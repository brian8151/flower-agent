from pydantic import BaseModel, Field
from typing import List, Optional


class DataItem(BaseModel):
    features: List[float] = Field(..., description="Prediction data features")


class PredictionRequest(BaseModel):
    domain_type: str = Field(..., alias="domainType", description="data seed domain type")
    workflow_trace_id: str = Field(..., alias="workflowTraceId", description="workflow trace id")
    data: List[DataItem] = Field(..., alias="data", description="prediction data list")


class PredictionWithWeightReq(BaseModel):
    domain_type: str = Field(..., alias="domainType", description="data seed domain type")
    workflow_trace_id: str = Field(..., alias="workflowTraceId", description="workflow trace id")
    data: List[DataItem] = Field(..., alias="data", description="prediction data list")
    weights: Optional[str] = Field(..., alias="weights", description="model weights")