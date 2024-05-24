from src.repository.db.db_connection import DBConnection


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
    max_version = result[0]
    sql = """
    INSERT INTO agent_local_model_history (workflow_trace_id, name, model_weights, version) VALUES ('{}', '{}', '{}','{}')""".format(workflow_trace_id, name, model_weights, max_version+1)
    DBConnection.execute_update(sql)
