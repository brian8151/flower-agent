from src.repository.db.db_connection import DBConnection
from src.util import log
logger = log.init_logger()

def create_local_model_historical_records(workflow_trace_id, name, model_weights):
    """
    create data process status to the database.

    Parameters:
        name (str): The name of model.
        workflow_trace_id (str): workflow trace id.
        model_weights (str): local model weights.
    """
    max_version_sql = """
        SELECT IFNULL(MAX(version), 0) FROM agent_local_model_history WHERE workflow_trace_id = %s
        """
    result = DBConnection.execute_fetch_one(max_version_sql)
    max_version = result[0] +1
    logger.info("create_local_model_historical_records max_version: {0}".format(max_version))
    sql = """
    INSERT INTO agent_local_model_history (workflow_trace_id, name, model_weights, version) VALUES ('{}', '{}', '{}','{}')""".format(workflow_trace_id, name, model_weights, max_version)
    DBConnection.execute_update(sql)
    logger.info("created local model historical records, name: {0}, workflow_trace_id: {1}, max_version: {2}".format(name, workflow_trace_id, max_version))
