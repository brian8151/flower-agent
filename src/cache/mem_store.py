import sqlite3
from src.util import log
logger = log.init_logger()

# Global in-memory database connection and cursor (to preserve data during server runtime)
def setup_mem_store():
    try:
        conn = sqlite3.connect(':memory:')
        cursor = conn.cursor()

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
    except sqlite3.Error as e:
        logger.error("SQLite error: %s", e)
    except Exception as e:
        logger.error("Unexpected error: %s", e)
    finally:
        if conn:
            conn.close()


# get model data
def get_model(name: int):
    conn = sqlite3.connect(':memory:')
    cursor = conn.cursor()
    cursor.execute('''SELECT definition FROM model WHERE name=?''', (name,))
    row = cursor.fetchone()
    conn.close()
    return {"model": row}


# get weight data
def get_weight(weight_id: int):
    conn = sqlite3.connect(':memory:')
    cursor = conn.cursor()
    cursor.execute('''SELECT weights FROM weight WHERE id=?''', (weight_id,))
    row = cursor.fetchone()
    conn.close()
    return {"weight": row}


# get weight data by model name
def get_weight_by_model(model_name: str):
    conn = sqlite3.connect(':memory:')
    cursor = conn.cursor()
    cursor.execute('''SELECT weight.weights 
                      FROM weight 
                      JOIN model ON weight.model_id = model.id 
                      WHERE model.model_name = ?''', (model_name,))
    row = cursor.fetchone()
    conn.close()
    if row:
        return {"weight": row[0]}
    else:
        return {"weight": ""}


# add weight data
def add_weight(model_id: int, weights: bytes):
    conn = sqlite3.connect(':memory:')
    cursor = conn.cursor()
    cursor.execute('''INSERT INTO weight (model_id, weights) VALUES (?, ?)''', (model_id, weights))
    conn.commit()
    conn.close()
    return {"message": "Weight added successfully"}


# add model data
def add_model(model_name: str, definition: bytes):
    conn = sqlite3.connect(':memory:')
    cursor = conn.cursor()
    cursor.execute('''INSERT INTO model (name, definition) VALUES (?, ?)''', (model_name, definition))
    conn.commit()
    conn.close()
    return {"message": "Model added successfully"}
