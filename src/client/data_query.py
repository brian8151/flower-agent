from random import random


class DataQuery:
    """ Class for Data Query service """
    def __init__(self):
        # Simulate a simple in-memory "database"
        self.data_storage = []
        self.results_storage = []

    def save_prediction_results(self, results):
        # Add "is_correct" field to each result
        for result in results:
            result["is_correct"] = random.choice(["Y", "N"])
        self.results_storage.extend(results)

    def get_prediction_results(self):
        # Retrieve saved prediction results
        return self.results_storage[:]

    def get_payment_data(self):
        # Each tuple corresponds to a row from the payment data table shown in your image
        return [
                (20604, -390.13, 0.76, 1),
                (21303, -1688.11, 0.88, 5),
                (31453, 695.59, 0.81, 7),
                (84606, -2302.61,0.79, 0),
                (80281, 2697.89, 0.92, 1),
                (12894, -2680.52, 0.999, 9),
                (30278, 565.45, 0.89, 2),
                (80323, -132.23,0.96, 1),
                (75689, 137.97, 0.78,3),
                (88564, -3933.18,0.88, 10),
                (50081, -1501.62, 0.87,10),
                (29561, 407.77,0.996, 10),
                (16321, -2495.47, 0.987, 10),
                (59563, 4164.26,0.927, 7),
                (51476,1187.14, 0.965, 5),
                (91007,-4717.40, 0.884, 8),
                (45000, 1200.00, 0.95, 2),
                (59563, 4164.26,0.927, 7),
                (71476, 1179.14, 0.925,9),
                (81007,-4762.40, 0.814, 2)
        ]