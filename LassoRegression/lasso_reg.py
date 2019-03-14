import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from google.cloud import bigquery
from sklearn.linear_model import Lasso
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import PolynomialFeatures
from joblib import Parallel, dump, load
from json import dumps
import datetime
import os
import sys


class Model():
    def __init__(self, model_id='lasso_model', model_ext='.joblib', model_compress=0):
        self._id = model_id
        self._ext = model_ext
        self.update_paths()
        self._compress = model_compress
        self._target = None
        self._features = None
        self._model = None
        model_loaded = False
        while not model_loaded:
            if os.path.exists(self.path_meta):
                old_model_meta = load(self.path_meta)
                print(old_model_meta)

                self._target = set(old_model_meta['target'])
                self._features = set(old_model_meta['features'])
                self._compress = old_model_meta['compress']
                self._model = load(self.path)

                model_loaded = True
            else:
                model_loaded = True

    def __del__(self):
        del self._model

    def train(self, target, train_data):
        if self._model is not None:
            print("Warning: Existing model will be overwritten.")
            response = ''
            while response != 'y' and response != 'n':
                response = input("Continue? (Y/n) > ").lower()
            if response == 'n':
                return
        self._target = set([target])
        self._features = train_data.columns - self._target

        model_loaded = False
        while not model_loaded:
            if os.path.exists(self.path_meta):
                old_model_meta = load(self.path_meta)
                if set(old_model_meta['features']) != self.features or set(old_model_meta['target']) != self.target:
                    conflict_resolved = False
                    while not conflict_resolved:
                        print(
                            "Existing model with conflicting features or target '" + self.id + "' exists.")
                        result_overwrite = ''
                        while result_overwrite != 'y' and result_overwrite != 'n':
                            result_overwrite = input(
                                "Overwrite? (Y/n) > ").lower()
                        if result_overwrite == 'y':
                            os.remove(self.path_meta)
                            os.remove(self.path)
                            conflict_resolved = True
                        else:
                            result_rename = ''
                            while result_rename != 'y' and result_rename != 'n':
                                result_rename = input(
                                    "Give new model id? (Y/n) > ").lower()
                            if result_rename == 'y':
                                self.id = input("New model id: > ")
                            else:
                                raise Exception(
                                    "Model creation failed - conflicting '"+self.id+"' already exists.")
                else:
                    conflict_resolved = True
                    model_loaded = True
            else:
                model_loaded = True

        self._model = Lasso(alpha=0.1)
        self._model.fit(train_data.data.loc[:,self.features], train_data.data.loc[:, self.target])

        target_predicted = self._model.predict(train_data.data.loc[:,self.features])

        rmse = np.sqrt(mean_squared_error(train_data.data.loc[:, self.target], target_predicted))
        r2 = r2_score(train_data.data.loc[:, self.target], target_predicted)

        return {'rmse':rmse, 'r2':r2}

    def test(self, test_data):
        test_predicted = self._model.predict(test_data.data.loc[:,self.features])

        rmse = np.sqrt(mean_squared_error(test_data.data.loc[:, self.target], test_predicted))
        r2 = r2_score(test_data.data.loc[:, self.target], test_predicted)

        return {'rmse':rmse, 'r2':r2}

    def dump(self):
        if self.meta is None:
            return False

        dump(self.meta, self._path_meta)
        dump(self._model, self._path, compress=self._compress)
        return True

    @property
    def path(self):
        if not self._path:
            self.update_paths()
        return self._path

    @property
    def path_meta(self):
        if not self._path_meta:
            self.update_paths()
        return self._path_meta

    @property
    def meta(self):
        if self._model is None or self._target is None or self._features is None:
            return None
        return dict(
            id=self._id,
            ext=self._ext,
            compress=self._compress,
            target=list(self._target),
            features=list(self._features)
        )

    def update_paths(self):
        self._path = os.path.join(os.path.dirname(
            os.path.abspath(__file__)), 'model/', self._id+self._ext)
        self._path_meta = os.path.join(os.path.dirname(
            os.path.abspath(__file__)), 'model/', self._id+'.json')

    @property
    def id(self):
        return self._id

    @id.setter
    def id(self, _id):
        self._id = _id
        self.update_paths()

    @property
    def ext(self):
        return self._ext

    @ext.setter
    def ext(self, _ext):
        self._ext = _ext
        self.update_paths()

    @property
    def compress(self):
        return self._compress

    @property
    def target(self):
        return self._target

    @property
    def features(self):
        return self._features

    @property
    def model(self):
        return self._model


class Dataset():
    def __init__(self, table_name='train', data_id='lasso_data', data_ext='.joblib.gz', data_compress=4, columns=None, max_size=None, data_where=None):
        self._id = data_id
        self._ext = data_ext
        self.update_paths()
        self._compress = data_compress
        self._max_size = max_size
        self._columns = set(columns)
        self._data = None
        self._table_name = table_name
        self._data_where = None
        data_loaded = False
        while not data_loaded:
            if os.path.exists(self.path_meta):
                old_data_meta = load(self.path_meta)
                if set(old_data_meta['columns']) != self.columns:
                    conflict_resolved = False
                    while not conflict_resolved:
                        print(
                            "Existing dataset with conflicting columns '" + self.id + "' exists.")
                        result_overwrite = ''
                        while result_overwrite != 'y' and result_overwrite != 'n':
                            result_overwrite = input(
                                "Overwrite? (Y/n) > ").lower()
                        if result_overwrite == 'y':
                            os.remove(self.path_meta)
                            os.remove(self.path)
                        else:
                            result_rename = ''
                            while result_rename != 'y' and result_rename != 'n':
                                result_rename = input(
                                    "Give new data id? (Y/n) > ").lower()
                            if result_rename == 'y':
                                data_id = input("New data id: > ")
                            else:
                                raise Exception(
                                    "Dataset creation failed - conflicting '"+self.id+"' already exists.")
                else:
                    self._data = load(self.path)
                    data_loaded = True
            else:
                self.load_new_data()
                data_loaded = True

    def __del__(self):
        del self._data

    def load_new_data(self):
        client = bigquery.Client()

        gsod_dataset_ref = client.dataset(
            'gsod_copy', project='to-hail-or-not-to-hail')
        gsod_dset = client.get_dataset(gsod_dataset_ref)
        gsod_clean = client.get_table(gsod_dset.table(self._table_name))
        if self._max_size is not None:
            self._data = client.list_rows(
                gsod_clean, max_results=self._max_size).to_dataframe()
        else:
            self._data = client.list_rows(gsod_clean).to_dataframe()
        if self._data_where is not None:
            self._data = self._data.loc[self._data.index.map(self._data_where),self._columns].dropna()
        else:
            self._data = self._data.loc[:, self._columns].dropna()
        if 'mo' in self._columns:
            print("'mo' present")
            self._data = self._data.astype({'mo':'int64'})

    def dump(self):
        if self.meta is None:
            return False
        dump(self.meta, self._path_meta)
        dump(self._data, self._path, compress=self._compress)
        return True

    @property
    def path(self):
        if not self._path:
            self.update_paths()
        return self._path

    @property
    def path_meta(self):
        if not self._path_meta:
            self.update_paths()
        return self._path_meta

    @property
    def meta(self):
        if self._data is None:
            return None
        return dict(
            table_name=self._table_name,
            id=self.id,
            ext=self.ext,
            compress=self._compress,
            columns=list(self._columns)
        )
        

    def update_paths(self):
        self._path = os.path.join(os.path.dirname(
            os.path.abspath(__file__)), 'data/', self._id+self._ext)
        self._path_meta = os.path.join(os.path.dirname(
            os.path.abspath(__file__)), 'data/', self._id+'.json')

    @property
    def id(self):
        return self._id

    @id.setter
    def id(self, _id):
        self._id = _id
        self.update_paths()

    @property
    def ext(self):
        return self._ext

    @ext.setter
    def ext(self, _ext):
        self._ext = _ext
        self.update_paths()

    @property
    def compress(self):
        return self._compress

    @property
    def columns(self):
        return self._columns

    @property
    def data(self):
        return self._data


class LassoReg():
    def __init__(self, max_rows=1000000, data_columns=['mo','temp', 'dewp', 'slp', 'stp', 'visib', 'wdsp', 'max', 'min', 'altitude', 'longitude', 'latitude', 'prcp'], data_where=None,target='prcp', model_id='lasso_model', model_ext='.joblib', model_compress=0, data_id='lasso_data', data_ext='.joblib.gz', data_compression=4):
        self.train_data = Dataset(data_id=data_id+"_train", data_ext=data_ext,
                                  data_compress=data_compression, columns=data_columns, max_size=max_rows,data_where=data_where)
        self.test_data = Dataset(table_name="test", data_id=data_id+"_test", data_ext=data_ext,
                                 data_compress=data_compression, columns=data_columns, max_size=int(max_rows/10),data_where=data_where)
        self.model = Model(model_id=model_id, model_ext=model_ext, model_compress=model_compress)
        self.target = target
        self.train_results = None
        self.test_results = None

    def train(self):
        self.train_results = self.model.train(self.target, self.train_data)

    def test(self):
        self.test_results = self.model.test(self.test_data)

    def results(self):
        if self.train_results is not None:
            print("Train Results:")
            print(self.format_results(self.train_results))
        if self.test_results is not None:
            print("Test Results:")
            print(self.format_results(self.test_results))
        
    def format_results(self,results):
        return "\tRMSE: {}\n\tR-Squared: {}\n".format(results['rmse'],results['r2'])

    def save(self):
        self.train_data.dump()
        print("Train data saved successfully.")
        self.test_data.dump()
        print("Test data saved successfully.")
        self.model.dump()
        print("Model saved successfully.")


if __name__ == '__main__':
    data_filter = None#lambda r: r.loc['country'] == 'UK'
    pm = LassoReg(max_rows=100000, model_id='lasso_model_alpha-0.1_10000', target='temp',
                 data_id='lasso_data_alpha-0.1_10000', data_compression=0, data_ext='.joblib', data_where=data_filter)
    pm.train()
    pm.test()
    pm.results()
    print(pm.model.model.coef_)
    pm.save()
