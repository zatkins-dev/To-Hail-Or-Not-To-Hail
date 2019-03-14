import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from google.cloud import bigquery
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import PolynomialFeatures
from joblib import Parallel, dump, load
from json import dumps
import datetime
import os
import sys
from functools import reduce
import easygui 

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
        self._target = set([target])
        self._features = train_data.columns - self._target
        self._degree = degree
        self._poly_features = PolynomialFeatures(degree=self._degree)

        data_poly = self._poly_features.fit_transform(
            train_data.data.loc[:, self.features])

        self._model = LinearRegression()
        self._model.fit(data_poly, train_data.data.loc[:, self.target])

        target_predicted = self._model.predict(data_poly)

        rmse = np.sqrt(mean_squared_error(
            train_data.data.loc[:, self.target], target_predicted))
        r2 = r2_score(train_data.data.loc[:, self.target], target_predicted)

        return {'RMSE':rmse, 'R-Squared':r2}

    def test(self, test_data):
        test_poly = self._poly_features.transform(
            test_data.data.loc[:, self.features])
        test_predicted = self._model.predict(test_poly)

        rmse = np.sqrt(mean_squared_error(
            test_data.data.loc[:, self.target], test_predicted))
        r2 = r2_score(test_data.data.loc[:, self.target], test_predicted)

        return {'RMSE':rmse, 'R-Squared':r2}

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
    def __init__(self, table_name='train',columns=None, max_size=None, data_where=None):
        self._max_size = max_size
        self._columns = set(columns)
        self._data = None
        self._table_name = table_name
        self._data_where = None
        # data_loaded = False
        # while not data_loaded:
        #     if os.path.exists(self.path_meta):
        #         old_data_meta = load(self.path_meta)
        #         if set(old_data_meta['columns']) != self.columns:
        #             conflict_resolved = False
        #             while not conflict_resolved:
        #                 print(
        #                     "Existing dataset with conflicting columns '" + self.id + "' exists.")
        #                 result_overwrite = ''
        #                 while result_overwrite != 'y' and result_overwrite != 'n':
        #                     result_overwrite = input(
        #                         "Overwrite? (Y/n) > ").lower()
        #                 if result_overwrite == 'y':
        #                     os.remove(self.path_meta)
        #                     os.remove(self.path)
        #                 else:
        #                     result_rename = ''
        #                     while result_rename != 'y' and result_rename != 'n':
        #                         result_rename = input(
        #                             "Give new data id? (Y/n) > ").lower()
        #                     if result_rename == 'y':
        #                         data_id = input("New data id: > ")
        #                     else:
        #                         raise Exception(
        #                             "Dataset creation failed - conflicting '"+self.id+"' already exists.")
        #         else:
        #             self._data = load(self.path)
        #             data_loaded = True
        #     else:
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

    @property
    def columns(self):
        return self._columns

    @property
    def data(self):
        return self._data


class PolyReg():
    def __init__(self, model_id, model_ext='.joblib', path=None,max_rows=100000, data_columns=['mo','temp', 'dewp', 'slp', 'stp', 'visib', 'wdsp', 'max', 'min', 'altitude', 'longitude', 'latitude', 'prcp'], data_where=None,target='temp', model_compress=0):
        self.train_data = Dataset(columns=data_columns, max_size=max_rows,data_where=data_where)
        self.test_data = Dataset(table_name="test",columns=data_columns, max_size=int(max_rows/10),data_where=data_where)
        self.model = None
        self.target = target
        self.train_results = None
        self.test_results = None
        self.model_path = path
        if self.model_path and os.path.exists(self.model_path):
            raise Warning(f"Warning: Model already exists at: \n'\t{self.model_path}'.\n.")

    def train(self,degree=2):
        if self.model is not None:
            print("Warning: Existing model will be overwritten.")
            response = ''
            while response != 'y' and response != 'n':
                response = input("Continue? (Y/n) > ").lower()
            if response == 'n':
                return
        self.model = Model()
        self.train_results = self.model.train(self.target, self.train_data, degree=degree)

    def test(self):
        self.test_results = self.model.test(self.test_data)

    def results(self):
        print("Train Results:")
        print(self.format_results(self.train_results))
        print("Test Results:")
        print(self.format_results(self.test_results))
        
    def format_results(self,results):
        if results is None:
            return '    No Results.\n'
        return reduce(lambda s1,s2: s1+s2,["    {0}: {1}\n".format(kv[0],kv[1]) for kv in results.items()])
    
    @classmethod
    def generate_model_path(cls, id, ext='.joblib', dirpath=os.path.dirname(os.path.abspath(__file__))):
        return os.path.join(dirpath, 'model/', id+ext)

    @classmethod
    def load_model_from_id(cls,id,ext='.joblib',dir=None):
        if dir is None:
            path = cls.generate_model_path(id) 
        else:
            path = os.path.join(dir, id+ext)
        if os.path.exists(path):
            pm = load(path)
            pm.model_path = path
            return pm
        else:
            return None
    
    @classmethod
    def load_model_from_path(cls,path):
        if os.path.exists(path):
            pm = load(path)
            pm.model_path = path
            return pm
        else:
            return None

    def save(self):
        if self.model_path is None:
            self.save_as(self.model_path)
        else:
            dump(self, self.model_path)
    
    def save_as(self,path):
        print("Save as...")
        if path is not None:
            self.model_path = path
            if not os.path.exists(self.model_path): 
                self.save()
                return
            print("Warning: Existing file will be overwritten.")
            response = ''
            while response != 'y' and response != 'n':
                response = input("Continue? (Y/n) > ").lower()
            if response == 'y':
                os.remove(path)
                self.save()
        else:
            while True:
                self.model_path = easygui.filesavebox(msg='Save as..',title='Save PolyReg Instance',default="*.joblib",filetypes=["*.joblib"])
                if self.model_path is None: return
                if not os.path.exists(self.model_path): break
            self.save()

    
if __name__ == '__main__':
    pm = PolyReg.load_model_from_id('test_model')
    if pm is None:
        pm = PolyReg('test_model',max_rows=None)
    pm.train()
    pm.test()
    pm.results()
    pm.save()
    pm.save_as(PolyReg.generate_model_path('test_model'))

