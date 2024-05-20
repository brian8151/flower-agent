from http.client import HTTPException
from typing import List

from fastapi import APIRouter
from src.util import log
from src.datamodel.data_item import PredictionRequest

logger = log.init_logger()
flower_router = APIRouter()


@flower_router.post("/predict-data")
async def receive_data(request: PredictionRequest):
    try:
        logger.info(f"Domain Type: {request.domain_type}")
        logger.info(f"Workflow Trace ID: {request.workflowtraceId}")
        for item in request.data:
            logger.info(f"Received data: {item.features}")
        return {"status": "success", "data_received": len(request.data)}
    except Exception as e:
        logger.error(f"Error processing data: {e}")
        raise HTTPException(status_code=500, detail="Internal Server Error")
