import numpy as np

from src.client.data_query import DataQuery
import pandas as pd
from src.util import log
logger = log.init_logger()

class DataProcessor:
    """ Class for Data Processor service """
    def __init__(self):
        self.db_query = DataQuery()

    # def prepare(self, data):
    #     d = pd.DataFrame(data)
    #     d = d.to_numpy()
    #     d = np.asarray(d).astype('float32')
    #     return d

    # def prepare(self, data):
    #     d = pd.DataFrame(data)
    #     X = d.iloc[:, :4]  # All four columns as features if truly no labels are required
    #     return X.to_numpy().astype('float32')

    def prepare(self, data):
        d = pd.DataFrame(data)
        X = d.iloc[:, :3]  # First three columns as features
        Y = d.iloc[:, 3]  # Fourth column as label
        X = X.to_numpy().astype('float32')
        Y = Y.to_numpy().astype('float32')
        return X, Y

    def fetch_and_prepare_payment_data(self):
        # Fetch the raw data
        data_raw = self.db_query.get_payment_data()
        X, Y = self.prepare(data_raw)
        logger.info(f"Prepared data shapes - X: {X.shape}, Y: {Y.shape}")
        logger.info(f"Sample data - X: {X[:5]}, Y: {Y[:5]}")
        return X, Y


    def process_results(self, predicted_data):
        x = np.array([entry['result'] for entry in predicted_data]).reshape(-1, 1)
        is_correct_req = [entry['is_correct'] for entry in predicted_data]

        y = []
        for result, correct in zip(x.flatten(), is_correct_req):
            if correct == "Y":
                y.append(0 if result > 73.0 else 1)
            else:
                y.append(1 if result > 73.0 else 0)
        y = np.array(y)
        return x, y

    def save_prediction_results(self, y_hat):
        results = [{"result": float(100.0 * y)} for y in y_hat.flatten()]
        self.db_query.save_prediction_results(results)
        return results

    def get_fit_data(self):
        predicted_data = self.db_query.get_prediction_results()
        x, y = self.process_results(predicted_data)
        return x, y