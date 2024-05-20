from pydantic import BaseModel
from typing import List


class DataItem(BaseModel):
    features: List[float]


class PredictionRequest(BaseModel):
    domain_type: str
    workflowtraceId: str
    data: List[DataItem]