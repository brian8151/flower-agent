from http.client import HTTPException
from typing import List

from fastapi import APIRouter

from src.service.moder_runner_service import ModelRunner
from src.util import log
from src.datamodel.data_item import PredictionRequest

logger = log.init_logger()
flower_router = APIRouter()


@flower_router.post("/predict-data")
async def receive_data(request: PredictionRequest):
    try:
        logger.info(f"Domain Type: {request.domain_type}")
        logger.info(f"Workflow Trace ID: {request.workflow_trace_id}")
        logger.info(f"data_received: {len(request.data)}")
        model_runner = ModelRunner()
        data_req = model_runner.run_mode_prediction(request.workflow_trace_id, request.domain_type, request.data)
        for item in request.data:
            logger.info(f"Received data: {item.features}")
        return {"status": "success", "data_received": len(request.data), "predictions": data_req}
    except Exception as e:
        logger.error(f"Error processing data: {e}")
        raise HTTPException(status_code=500, detail="Internal Server Error")
