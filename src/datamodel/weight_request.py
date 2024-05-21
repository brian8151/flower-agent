from pydantic import BaseModel, Field


class WeightRequest(BaseModel):
    model: str = Field(..., alias="model", description="model json format")
