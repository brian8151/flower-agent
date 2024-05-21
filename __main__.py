import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from src.cache.mem_store import close_connection
from src.router.agent_flower_router import flower_router
import logging
from src.service.model_operator import ModelOperator
import sys
from os.path import dirname as opd, realpath as opr
_BASEDIR_ = opd(opr(__file__))
sys.path.append(_BASEDIR_)
from dotenv import load_dotenv
load_dotenv(dotenv_path=_BASEDIR_+"/app.env")

def create_app():
    logging.getLogger("uvicorn.error").setLevel(logging.INFO)
    app = FastAPI()

    @app.on_event("startup")
    def on_startup():
        try:
            model_ops = ModelOperator()
            model_ops.initial_mem_store()
            logging.info("Aikya FL Client started and is listening on http://0.0.0.0:7000")
        except Exception as e:
            logging.error("Unexpected error: %s", e)

    @app.on_event("shutdown")
    def on_shutdown():
        close_connection()

    # Set up CORS
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # Allows all origins
        allow_credentials=True,
        allow_methods=["*"],  # Allows all methods
        allow_headers=["*"],  # Allows all headers
    )
    app.include_router(flower_router, prefix="/flwr/api")
    return app


# Legacy mode
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    app = create_app()
    uvicorn.run(app, host="0.0.0.0", port=7000)
