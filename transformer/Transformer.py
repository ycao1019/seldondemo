import numpy as np
import joblib

class Transformer(object):

    def __init__(self):
        self.transformer = joblib.load('input_transformer.pkl')

    def transform_input(self, X, feature_names):
        X = np.array(X)
        return self.transformer.transform(X)