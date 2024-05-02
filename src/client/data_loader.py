import pandas as pd

class DataLoader:
    def __init__(self, filepath):
        self.filepath = filepath

    def load_data(self):
        print("DataLoader file:", self.filepath)
        data = pd.read_csv(self.filepath)
        features = data.drop('is_fraudulent', axis=1)
        labels = data['is_fraudulent']
        return features, labels