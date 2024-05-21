import sqlite3
from src.util import log

logger = log.init_logger()

# Global connection variable
conn = None
cursor = None


def get_connection():
    global conn, cursor
    if conn is None:
        conn = sqlite3.connect(':memory:', check_same_thread=False)
        cursor = conn.cursor()
        logger.info("Database connection opened.")
    return conn, cursor


def close_connection():
    global conn
    if conn:
        conn.close()
        logger.info("Database connection closed.")
        conn = None


# Global in-memory database connection and cursor (to preserve data during server runtime)
def setup_mem_store():
    try:
        conn, cursor = get_connection()
        # Create the model table
        cursor.execute('''CREATE TABLE model (
                            id INTEGER PRIMARY KEY,
                            name TEXT,
                            definition BLOB
                          )''')

        # Create the weight table
        cursor.execute('''CREATE TABLE weight (
                            id INTEGER PRIMARY KEY,
                            model_id INTEGER,
                            weights BLOB,
                            FOREIGN KEY (model_id) REFERENCES model_table(id)
                          )''')

        conn.commit()
        # Query to list all tables
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tables = cursor.fetchall()
        logger.info("==========>  Tables in the database: %s  <============", tables)
    except sqlite3.Error as e:
        logger.error("SQLite error: %s", e)
    except Exception as e:
        logger.error("Unexpected error: %s", e)


# get model data
def get_model(name: int):
    try:
        conn, cursor = get_connection()
        cursor.execute('''SELECT definition FROM model WHERE name=?''', (name,))
        row = cursor.fetchone()
        if row:
            return row[0]
        else:
            raise ValueError(f"Model {name} not found in the database")
    except Exception as e:
        logger.error("get_model - Unexpected error: %s", e)


# get weight data
def get_weight(weight_id: int):
    try:
        conn, cursor = get_connection()
        cursor.execute('''SELECT weights FROM weight WHERE id=?''', (weight_id,))
        row = cursor.fetchone()
        return {"weight": row}
    except Exception as e:
        logger.error("get_weight_by_model - Unexpected error: %s", e)


# get weight data by model name
def get_weight_by_model(model_name: str):
    try:
        conn, cursor = get_connection()
        cursor.execute('''SELECT weight.weights 
                          FROM weight 
                          JOIN model ON weight.model_id = model.id 
                          WHERE model.model_name = ?''', (model_name,))
        row = cursor.fetchone()
        if row:
            return {"weight": row[0]}
        else:
            return {"weight": ""}
    except Exception as e:
        logger.error("get_weight_by_model - Unexpected error: %s", e)


# add weight data
def add_weight(model_id: int, weights: bytes):
    try:
        conn, cursor = get_connection()
        cursor.execute('''INSERT INTO weight (model_id, weights) VALUES (?, ?)''', (model_id, weights))
        conn.commit()
        return {"message": "Weight added successfully"}
    except Exception as e:
        logger.error("add_weight - Unexpected error: %s", e)


# add model data
def create_model(model_name: str, definition: bytes):
    logger.info("start adding model: {0}".format(model_name))
    try:
        conn, cursor = get_connection()
        cursor.execute('''INSERT INTO model (name, definition) VALUES (?, ?)''', (model_name, definition))
        conn.commit()
        logger.info("complete add model: {0}".format(model_name))
        return {"message": "Model added successfully"}
    except Exception as e:
        logger.error("add_model - Unexpected error: %s", e)
