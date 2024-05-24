from src.repository.db.db_connection import DBConnection


def create_local_model_historical_records(workflow_trace_id, name, model_weights):
    """
    create data process status to the database.

    Parameters:
        name (str): The name of model.
        workflow_trace_id (str): workflow trace id.
        model_weights (str): local model weights.
    """
    sql = """
    INSERT INTO agent_local_model_history (workflow_trace_id, name, model_weights, version) 
    VALUES ('{}', '{}', '{}', (SELECT IFNULL(MAX(version), 0) + 1 FROM agent_local_model_history WHERE name = '{}'))
    """.format(workflow_trace_id, name, model_weights, name)
    DBConnection.execute_update(sql)
