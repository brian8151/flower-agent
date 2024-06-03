from src.repository.db.db_connection import DBConnection
from src.util import log

logger = log.init_logger()


def get_model_track_record(domain):
    """
    Retrieve the global model track for the given domain from the database.

    Parameters:
        domain (str): The domain for which to retrieve the global model track.

    Returns:
        tuple: A tuple containing the model definition, model version, local model weights,
               local weights version, global model weights, and global weights version.
               Returns an empty tuple if no record is found.
    """
    sql = (
        """select definition, model_version, local_model_weights, local_weights_version, global_model_weights, global_weights_version from agent_model_records where name='{}'"""
        .format(domain))
    result = DBConnection.execute_query(sql)
    if result:
        return result[0][0], result[0][1], result[0][2], result[0][3], result[0][4], result[0][5]
    return ()


def update_global_model_track(name, global_model_weights, global_weights_version):
    """
    Update the status of a data process in the database.

    Parameters:
        name (str): The name of model.
        global_model_weights (str): The global model weights
        global_weights_version (str): The global model weights version
    """

    sql = """UPDATE agent_model_records SET global_model_weights='{}' and global_weights_version='{}' WHERE name='{}'""".format(
        global_model_weights, global_weights_version, name)
    print(f"SQL Query: {sql}")
    DBConnection.execute_update(sql)


def update_local_model_track(name, local_model_weights, local_weights_version):
    """
    Update the status of a data process in the database.

    Parameters:
        name (str): The name of model.
        local_model_weights (str): The local model weights
        local_weights_version (str): The local model weights version
    """

    sql = """UPDATE agent_model_records SET local_model_weights='{}' and local_weights_version='{}' WHERE name='{}'""".format(
        local_model_weights, local_weights_version, name)
    print(f"SQL Query: {sql}")
    DBConnection.execute_update(sql)


def create_model_track_records(name, definition, model_version, domain_type, local_model_weights,
                               local_weights_version):
    """
    create data process status to the database.

    Parameters:
        name (str): The name of model.
        definition (str): The model definition.
        model_version (str): The version of the model.
        domain_type (str): domain type such as payment.
        local_model_weights (str): weights.
        local_weights_version (int): version of weights.
    """
    sql = """insert into agent_model_records (name, definition, model_version, domain, local_model_weights, local_weights_version) VALUES('{}', '{}', '{}', '{}', '{}', '{}')""".format(
        name, definition, model_version, domain_type, local_model_weights, local_weights_version)
    DBConnection.execute_update(sql)


def get_model_track_record(domain):
    """
    Retrieve the global model track for the given domain from the database.

    Parameters:
        domain (str): The domain for which to retrieve the global model track.

    Returns:
        tuple: A tuple containing the model definition, model version, local model weights,
               local weights version, global model weights, and global weights version.
               Returns an empty tuple if no record is found.
    """
    sql = (
        """select definition, model_version, local_model_weights, local_weights_version, global_model_weights, global_weights_version from agent_model_records where name='{}'"""
        .format(domain))
    result = DBConnection.execute_query(sql)
    if result:
        return result[0][0], result[0][1], result[0][2], result[0][3], result[0][4], result[0][5]
    return ()


def create_local_model_historical_records(workflow_trace_id, name, model_weights):
    """
    create data process status to the database.

    Parameters:
        name (str): The name of model.
        workflow_trace_id (str): workflow trace id.
        model_weights (str): local model weights.
    """
    max_version_sql = """SELECT IFNULL(MAX(version), 0) FROM agent_local_model_history WHERE name ='{}'""".format(name)
    result = DBConnection.execute_fetch_one(max_version_sql)
    max_version = result[0] + 1
    logger.info("create_local_model_historical_records max_version: {0}".format(max_version))
    sql = """
    INSERT INTO agent_local_model_history (workflow_trace_id, name, model_weights, version) VALUES ('{}', '{}', '{}','{}')""".format(
        workflow_trace_id, name, model_weights, max_version)
    DBConnection.execute_update(sql)
    logger.info(
        "created local model historical records, name: {0}, workflow_trace_id: {1}, max_version: {2}".format(name,
                                                                                                             workflow_trace_id,
                                                                                                             max_version))


def create_workflow_model_process(workflow_trace_id, event, status):
    sql = """INSERT INTO workflow_model_logs (workflow_trace_id, event, status)
             VALUES ('{}', '{}', '{}')""".format(workflow_trace_id, event, status)
    DBConnection.execute_query(sql)


def update_workflow_model_process(workflow_trace_id, event, status):
    sql = """UPDATE workflow_model_logs 
             SET status='{}' 
             WHERE workflow_trace_id='{}' AND event='{}'""".format(status, workflow_trace_id, event)
    DBConnection.execute_query(sql)
