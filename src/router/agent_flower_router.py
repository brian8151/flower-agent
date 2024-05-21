from http.client import HTTPException

from fastapi import APIRouter

from src.datamodel.predict_request import PredictionRequest, PredictionReq
from src.datamodel.weight_request import WeightRequest
from src.service.moder_runner_service import ModelRunner
from src.util import log

logger = log.init_logger()
flower_router = APIRouter()


@flower_router.post("/initial-weights")
async def get_weights(request: WeightRequest):
    try:
        model_runner = ModelRunner()
        weights = model_runner.get_model_weights_with_compression(request.model)
        return {"status": "success", "weights": weights}
    except Exception as e:
        logger.error(f"Error getting model weights: {e}")
        raise HTTPException(status_code=500, detail=f"Internal Server Error: {e}")


@flower_router.post("/predict")
async def predict(request: PredictionReq):
    try:
        logger.info(f"Domain Type: {request.domain_type}")
        logger.info(f"Workflow Trace ID: {request.workflow_trace_id}")
        logger.info(f"data_received: {len(request.data)}")
        model_runner = ModelRunner()
        data_req = model_runner.run_model_prediction(request.workflow_trace_id, request.domain_type, request.data)
        for item in request.data:
            logger.info(f"Received data: {item.features}")
        return {"status": "success", "data_received": len(request.data), "predictions": data_req}
    except Exception as e:
        logger.error(f"Error predict data: {e}")
        raise HTTPException(status_code=500, detail="Internal Server Error")


@flower_router.post("/predict-data")
async def predict_data(request: PredictionRequest):
    try:
        logger.info(f"Domain Type: {request.domain_type}")
        logger.info(f"Workflow Trace ID: {request.workflow_trace_id}")
        logger.info(f"data_received: {len(request.data)}")
        model_runner = ModelRunner()
        data_req = model_runner.run_model_prediction(request.workflow_trace_id, request.domain_type, request.data)
        for item in request.data:
            logger.info(f"Received data: {item.features}")
        return {"status": "success", "data_received": len(request.data), "predictions": data_req}
    except Exception as e:
        logger.error(f"Error predict data: {e}")
        raise HTTPException(status_code=500, detail="Internal Server Error")
