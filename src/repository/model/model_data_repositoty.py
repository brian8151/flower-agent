from src.repository.db.db_connection import DBConnection


def get_model_feature_record(domain, batch_id):
    """
    Constructs and executes a dynamic SQL query to retrieve records from the model_data_features table
    based on the specified domain and batch_id.

    Args:
        domain (str): The domain to filter the records.
        batch_id (str): The batch_id to filter the records.

    Returns:
        list: A list of tuples representing the records retrieved from the database.
    """
    # Construct the SQL query to retrieve db_table, id_field, and feature_field based on the domain
    sql = """
        SELECT db_table, id_field, feature_field 
        FROM model_data_features 
        WHERE domain='{0}' AND model='{1}' AND status='Active' 
        ORDER BY seq_num
    """.format(domain, domain)

    # Execute the query and fetch the results
    rows = DBConnection.execute_query(sql)

    if rows:
        # Extract the table name and id field from the first row
        db_table = rows[0][0]
        id_field = rows[0][1]

        # Extract the feature fields from the remaining rows
        feature_fields = [row[2] for row in rows]

        # Construct the SELECT clause by joining the feature fields
        select_fields = [id_field] + feature_fields
        select_clause = ", ".join(select_fields)

        # Construct the dynamic SQL query to retrieve records based on the batch_id
        dynamic_query = f"SELECT {select_clause} FROM {db_table} WHERE batch_id='{batch_id}'"

        # Print the dynamic query for debugging purposes
        print(dynamic_query)

        # Execute the dynamic query and fetch the results
        result = DBConnection.execute_query(dynamic_query)
        # Return the results
        return result
    else:
        print("No active features found for the specified domain and model.")

    return []