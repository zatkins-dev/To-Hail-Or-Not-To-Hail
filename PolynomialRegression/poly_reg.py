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
import os, sys


class Model():
    def __init__(self,model_id='poly_model',model_ext='.joblib',model_compress=0):
        self._id = model_id
        self._ext = model_ext
        self.update_paths()
        self._compress = model_compress
        self._target = None
        self._features = None
        self._poly_features = None
        self._model = None
        model_loaded = False
        while not model_loaded:
            if os.path.exists(self.path_meta):
                old_model_meta = load(self.path_meta)
                print(old_model_meta)
                
                self._target = set(old_model_meta['target'])
                self._features = set(old_model_meta['features'])
                self._compress = old_model_meta['compress']
                self._degree = old_model_meta['degree']
                self._model = load(self.path)

                self._poly_features = PolynomialFeatures(degree = self._degree)
                model_loaded = True

            else:
                model_loaded = True


    def __del__(self):
        del self._poly_features
        del self._model
        

    def train(self,target,train_data,degree=4):
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
                        print("Existing model with conflicting features or target '" + self.id + "' exists.")
                        result_overwrite = ''
                        while result_overwrite != 'y' and result_overwrite != 'n':
                            result_overwrite = input("Overwrite? (Y/n) > ").lower()
                        if result_overwrite == 'y':
                            os.remove(self.path_meta)
                            os.remove(self.path)
                            conflict_resolved = True
                        else:
                            result_rename = ''
                            while result_rename != 'y' and result_rename != 'n':
                                result_rename = input("Give new model id? (Y/n) > ").lower()
                            if result_rename == 'y':
                                self.id = input("New model id: > ")
                            else:
                                raise Exception(f"Model creation failed - conflicting '{self.id}' already exists.")
                else:
                    conflict_resolved = True
                    model_loaded = True
            else:
                model_loaded = True
        
        self._degree = degree
        self._poly_features = PolynomialFeatures(degree=self._degree)
        data_poly = self._poly_features.fit_transform(train_data.data.loc[:,self.features])
        self._poly_params = self._poly_features.get_params()

        self._model = LinearRegression()
        self._model.fit(data_poly,train_data.data.loc[:,self.target])

        target_predicted = self._model.predict(data_poly)

        rmse = np.sqrt(mean_squared_error(train_data.data.loc[:,self.target], target_predicted))
        r2 = r2_score(train_data.data.loc[:,self.target], target_predicted)

        print("Train Data Results")
        print(f"RMSE: {rmse}")
        print(f"R-Squared: {r2}")

    def dump(self):
        if self.meta is None:
            return False
        
        # meta_json = dumps(self.meta)
        # meta_file = open(self.path_meta, "w")
        # meta_file.write(meta_json)
        # meta_file.close()
        dump(self.meta,self._path_meta)
        dump(self._model,self._path,compress=self._compress)
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
            id = self._id,
            ext = self._ext,
            compress = self._compress,
            target = list(self._target),
            features = list(self._features),
            degree = self._degree,
        )

    def update_paths(self):
        self._path = os.path.join(os.path.dirname(os.path.abspath(__file__)),self._id+self._ext)
        self._path_meta = os.path.join(os.path.dirname(os.path.abspath(__file__)),self._id+'.json')

    @property
    def id(self):
        return self._id
    
    @id.setter
    def id(self,_id):
        self._id = _id
        self.update_paths()

    @property
    def ext(self):
        return self._ext
    
    @ext.setter
    def ext(self,_ext):
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
    def __init__(self,table_name='train',data_id='poly_data',data_ext='.joblib.gz',data_compress=4,columns=None,max_size=None):
        self._id = data_id
        self._ext = data_ext
        self.update_paths()
        self._compress = data_compress
        self._max_size = max_size
        self._columns = set(columns)
        self._data = None
        self._table_name = table_name
        data_loaded = False
        while not data_loaded:
            if os.path.exists(self.path_meta):
                old_data_meta = load(self.path_meta)
                print(old_data_meta)
                if set(old_data_meta['columns']) != self.columns:
                    conflict_resolved = False
                    while not conflict_resolved:
                        print("Existing dataset with conflicting columns '" + self.id + "' exists.")
                        result_overwrite = ''
                        while result_overwrite != 'y' and result_overwrite != 'n':
                            result_overwrite = input("Overwrite? (Y/n) > ").lower()
                        if result_overwrite == 'y':
                            os.remove(self.path_meta)
                            os.remove(self.path)
                        else:
                            result_rename = ''
                            while result_rename != 'y' and result_rename != 'n':
                                result_rename = input("Give new data id? (Y/n) > ").lower()
                            if result_rename == 'y':
                                data_id = input("New data id: > ")
                            else:
                                raise Exception(f"Dataset creation failed - conflicting '{self.id}' already exists.")
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

        gsod_dataset_ref = client.dataset('gsod_copy', project='to-hail-or-not-to-hail')
        gsod_dset = client.get_dataset(gsod_dataset_ref)
        gsod_clean = client.get_table(gsod_dset.table(self._table_name))
        if self._max_size is not None:
            self._data = client.list_rows(gsod_clean,max_results=self._max_size).to_dataframe()
        else:
            self._data = client.list_rows(gsod_clean).to_dataframe()
        self._data = self._data.loc[:,self._columns].dropna()

    def dump(self):
        if self.meta is None:
            return False
        
        # meta_json = dumps(self.meta)
        # meta_file = open(self.path_meta, "w")
        # meta_file.write(meta_json)
        # meta_file.close()

        dump(self.meta,self._path_meta)
        dump(self._data,self._path,compress=self._compress)
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
            id = self.id,
            ext = self.ext,
            compress = self._compress,
            columns = list(self._columns)
        )

    def update_paths(self):
        self._path = os.path.join(os.path.dirname(os.path.abspath(__file__)),self._id+self._ext)
        self._path_meta = os.path.join(os.path.dirname(os.path.abspath(__file__)), self._id+'.json')

    @property
    def id(self):
        return self._id
    
    @id.setter
    def id(self,_id):
        self._id = _id
        self.update_paths()

    @property
    def ext(self):
        return self._ext
    
    @ext.setter
    def ext(self,_ext):
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


class PolyReg():
    def __init__(self,max_rows=1000000,data_columns=['temp','dewp','slp','stp','visib','wdsp','max','min','altitude','longitude','latitude','prcp'],target='prcp',model_id='poly_model',model_ext='.joblib',model_compress=0,data_id='poly_data',data_ext='.joblib.gz',data_compression=4):
        self.data = Dataset(data_id=data_id,data_ext=data_ext,data_compress=data_compression,columns=data_columns,max_size=max_rows)
        self.model = Model(model_id=model_id,model_ext=model_ext,model_compress=model_compress)
        self.target = target

    def run(self):
        self.model.train(self.target,self.data)

    def save(self):
        self.data.dump()
        print("Data saved successfully.")
        self.model.dump()
        print("Model saved successfully.")



if __name__ == '__main__':
    pm = PolyReg(max_rows=1000000,model_id='poly_model_1000000',data_id='poly_data_1000000',data_compression=0,data_ext='.joblib')
    pm.run()
    pm.save()