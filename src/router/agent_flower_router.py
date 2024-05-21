from http.client import HTTPException

from fastapi import APIRouter

from src.datamodel.data_item import PredictionRequest
from src.datamodel.weight_request import WeightRequest
from src.service.moder_runner_service import ModelRunner
from src.util import log

logger = log.init_logger()
flower_router = APIRouter()


@flower_router.post("/get-initial-weights")
async def receive_data(request: WeightRequest):
    try:
        logger.info(f"model: {request.model}")
        model_runner = ModelRunner()
        weights = model_runner.get_model_weights(request.model)
        return {"status": "success", "weights": weights}
    except Exception as e:
        logger.error(f"Error get model weights: {e}")
        raise HTTPException(status_code=500, detail="Internal Server Error")

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
        logger.error(f"Error predict data: {e}")
        raise HTTPException(status_code=500, detail="Internal Server Error")
