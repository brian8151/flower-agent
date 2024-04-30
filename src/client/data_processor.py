import numpy as np

from src.client.data_query import DataQuery
import pandas as pd

class DataProcessor:
    """ Class for Data Processor service """
    def __init__(self):
        self.db_query = DataQuery()

    def prepare(self, data):
        d = pd.DataFrame(data)
        d = d.to_numpy()
        d = np.asarray(d).astype('float32')
        return d
    def fetch_and_prepare_payment_data(self):
        #for now
        data_raw = self.db_query.get_payment_data()
        data = self.prepare(data_raw)
        return data

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