from pydantic import BaseModel, Field
from typing import List


class DataItem(BaseModel):
    features: List[float] = Field(..., description="Prediction data features")


class PredictionRequest(BaseModel):
    domain_type: str = Field(..., alias="domainType", description="data seed domain type")
    workflow_trace_id: str = Field(..., alias="workflowTraceId", description="workflow trace id")
    data: List[DataItem] = Field(..., alias="data", description="prediction data list")
