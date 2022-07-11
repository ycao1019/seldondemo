import numpy as np
import joblib

class DemoModel(object):

    def __init__(self):
        self.model = joblib.load('model.pkl')

    def predict(self, X, features_names=None):
        X = np.array(X)
        return self.model.predict(X)