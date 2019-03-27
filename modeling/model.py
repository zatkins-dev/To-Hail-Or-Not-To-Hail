import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, explained_variance_score, mean_absolute_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures
import time

class Model():
    def __init__(self):
        self._target = None
        self._features = None
        self._poly_features = None
        self._model = None

    def train(self, target, train_data, degree=2,write=None):
        if self._model is not None:
            print("Warning: Existing model will be overwritten.")
            response = ''
            while response != 'y' and response != 'n':
                response = input("Continue? (Y/n) > ").lower()
            if response == 'n':
                return
        self._target = [target]
        self._features = train_data.columns
        self._features.remove(target)
        self._degree = degree
        target_predicted = None
        if self._degree is None or self._degree < 2:
            self._model = Pipeline([('regression',LinearRegression())])
        else:
            self._model = Pipeline([('kernel',PolynomialFeatures(degree=self._degree)), ('regression',LinearRegression())])

        print("      --> Training model...")
        if write: write("      --> Training model...\n")
        start = time.perf_counter()
        self._model.fit(train_data.data.loc[:, self.features],train_data.data.loc[:, self.target])
        cost = time.perf_counter() - start
        print("    --> Model trained ({} s).".format(cost))
        if write: write("    --> Model trained ({} s).".format(cost))

        print("    --> Predicting train data...")
        if write: write("    --> Predicting train data...\n")
        start = time.perf_counter()
        target_predicted = self._model.predict(train_data.data.loc[:, self.features])
        cost = time.perf_counter() - start
        print("    --> Train data predicted ({} s).".format(cost))
        if write: write("    --> Train data predicted ({} s).\n".format(cost))

        print("    --> Calculating train accuracy metrics...")
        if write: write("    --> Calculating train accuracy metrics...\n")
        start = time.perf_counter()
        me = mean_absolute_error(train_data.data.loc[:, self.target], target_predicted)
        var = explained_variance_score(train_data.data.loc[:, self.target], target_predicted)
        rmse = np.sqrt(mean_squared_error(train_data.data.loc[:, self.target], target_predicted))
        r2 = r2_score(train_data.data.loc[:, self.target], target_predicted)
        cost = time.perf_counter() - start
        print("    --> Train metrics calculated ({} s).".format(cost))
        if write: write("    --> Train metrics calculated ({} s).\n".format(cost))
        return {'R-Squared':r2, "Explained Variance":var, 'Root Mean Squared Error':rmse, 'Mean Absolute Error':me}

    def test(self,test_data,write=None):
        print("    --> Predicting test data...")
        if write: write("      --> Predicting test data...\n")
        start = time.perf_counter()
        test_predicted = self._model.predict(test_data.data.loc[:,self._features])
        cost = time.perf_counter() - start
        print("    --> Test data predicted ({} s).".format(cost))
        if write: write("      --> Test data predicted ({} s).\n".format(cost))
        
        print("    --> Calculating test accuracy metrics...")
        if write: write("      --> Calculating test accuracy metrics...\n")
        start = time.perf_counter()
        me = mean_absolute_error(test_data.data.loc[:, self.target], test_predicted)
        var = explained_variance_score(test_data.data.loc[:, self.target], test_predicted)
        rmse = np.sqrt(mean_squared_error(test_data.data.loc[:, self.target], test_predicted))
        r2 = r2_score(test_data.data.loc[:, self.target], test_predicted)
        cost = time.perf_counter() - start
        print("    --> Test metrics calculated ({} s).".format(cost))
        if write: write("    --> Test metrics calculated ({} s).\n".format(cost))
        return {'R-Squared':r2, "Explained Variance":var, 'Root Mean Squared Error':rmse, 'Mean Absolute Error':me}

    @property
    def target(self):
        return self._target

    @property
    def features(self):
        return self._features

    @property
    def model(self):
        return self._model

