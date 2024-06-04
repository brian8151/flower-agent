from http.client import HTTPException

from fastapi import APIRouter

from src.datamodel.model_request import PredictRequest, TraningRequest
from src.datamodel.weight_request import WeightRequest
from src.service.moder_runner_service import ModelRunner
from src.util import log

logger = log.init_logger()
agent_router = APIRouter()


@agent_router.post("/initial-weights")
async def initial_weights(request: WeightRequest):
    try:
        model_runner = ModelRunner()
        weights = model_runner.initial_weights(request.name, request.domain, request.version, request.model)
        return {"status": "success", "domain": request.domain, "weights": weights}
    except Exception as e:
        logger.error(f"Error getting model weights: {e}")
        raise HTTPException(status_code=500, detail=f"Internal Server Error: {e}")


@agent_router.get("/health")
async def health():
    try:
        return {"status": "success"}
    except Exception as e:
        logger.error(f"Error getting health: {e}")
        raise HTTPException(status_code=500, detail=f"Internal Server Error: {e}")


@agent_router.post("/get-model-weights")
async def get_model_weights(request: WeightRequest):
    try:
        model_runner = ModelRunner()
        weights = model_runner.get_model_weights_req(request.name, request.domain, request.version, request.model)
        return {"status": "success", "domain": request.domain, "weights": weights}
    except Exception as e:
        logger.error(f"Error getting model weights: {e}")
        raise HTTPException(status_code=500, detail=f"Internal Server Error: {e}")


@agent_router.post("/predict")
async def predict_data(request: PredictRequest):
    try:
        logger.info(f"Domain Type: {request.domain_type}")
        logger.info(f"Workflow Trace ID: {request.workflow_trace_id}")
        model_runner = ModelRunner()
        status, workflow_trace_id, n  = model_runner.run_model_predict(request.workflow_trace_id, request.domain_type, request.batch_id)
        return {"status": status, "workflow_trace_id": workflow_trace_id, "items": n}
    except Exception as e:
        logger.error(f"Error predict data: {e}")
        raise HTTPException(status_code=500, detail="Internal Server Error")


@agent_router.post("/training")
async def training_data(request: TraningRequest):
    try:
        logger.info(f"Domain Type: {request.domain_type}")
        logger.info(f"Workflow Trace ID: {request.workflow_trace_id}")
        model_runner = ModelRunner()
        workflow_trace_id, loss, num_examples, metrics  = model_runner.run_model_training(request.workflow_trace_id, request.domain_type, request.batch_id)
        return {"workflow_trace_id": workflow_trace_id, "loss": loss, "num_examples": num_examples, "metrics": metrics}
    except Exception as e:
        logger.error(f"Error predict data: {e}")
        raise HTTPException(status_code=500, detail="Internal Server Error")