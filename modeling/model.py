import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, r2_score, explained_variance_score, mean_absolute_error
from sklearn.preprocessing import PolynomialFeatures

class Model():
    def __init__(self):
        self._target = None
        self._features = None
        self._poly_features = None
        self._model = None

    def train(self, target, train_data, degree=4):
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
        self._poly_features = PolynomialFeatures(degree=self._degree)

        data_poly = self._poly_features.fit_transform(
            train_data.data.loc[:, self.features])

        self._model = Ridge(alpha=1e4)
        self._model.fit(data_poly, train_data.data.loc[:, self.target])

        target_predicted = self._model.predict(data_poly)

        me = mean_absolute_error(train_data.data.loc[:, self.target], target_predicted)
        var = explained_variance_score(train_data.data.loc[:, self.target], target_predicted)
        rmse = np.sqrt(mean_squared_error(
            train_data.data.loc[:, self.target], target_predicted))
        r2 = r2_score(train_data.data.loc[:, self.target], target_predicted)
        return {'R-Squared':r2, "Explained Variance":var, 'Root Mean Squared Error':rmse, 'Mean Absolute Error':me}

    def test(self, test_data):
        test_poly = self._poly_features.transform(
            test_data.data.loc[:, self.features])
        test_predicted = self._model.predict(test_poly)

        me = mean_absolute_error(test_data.data.loc[:, self.target], test_predicted)
        var = explained_variance_score(test_data.data.loc[:, self.target], test_predicted)
        rmse = np.sqrt(mean_squared_error(
            test_data.data.loc[:, self.target], test_predicted))
        r2 = r2_score(test_data.data.loc[:, self.target], test_predicted)
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
