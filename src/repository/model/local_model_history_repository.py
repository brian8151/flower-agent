from src.repository.db.db_connection import DBConnection


def create_local_model_historical_records(workflow_trace_id, name, model_weights):
    """
    create data process status to the database.

    Parameters:
        name (str): The name of model.
        workflow_trace_id (str): workflow trace id.
        model_weights (str): local model weights.
    """
    sql = """insert into agent_local_model_history (workflow_trace_id, name, model_weights) VALUES('{}', '{}', '{}')""".format(
        workflow_trace_id, name, model_weights)
    DBConnection.execute_update(sql)
