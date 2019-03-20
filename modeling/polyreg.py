import numpy as np
import pandas as pd
from sklearn.externals.joblib import dump, load
from statsmodels.stats.anova import anova_lm
import pickle
from functools import reduce
import easygui 
import os
from . import Dataset,Model

class PolyReg():
    def __init__(self, model_id, model_ext='.joblib', train_data_path=None,test_data_path=None,path=None,max_rows=100000, data_columns=['mo','temp', 'dewp', 'slp', 'stp', 'visib', 'wdsp', 'altitude', 'longitude', 'latitude', 'prcp'], data_where=None,target='temp', model_compress=0):
        self.model_id = None
        self.model_path = path
        self.model = Model()
        self.target = target
        self.train_results = None
        self.test_results = None
        if self.model_path is not None:
            self.model = Model()
            self.train_data = Dataset(columns=None)
            self.test_data = Dataset(columns=None)
            self.load_model_from_path(path)
            return
        print("  --> Loading Train Data")        
        if train_data_path is not None:
            self.train_data = Dataset(columns=None)
            self.train_data._data = load(train_data_path)
        else:
            self.train_data = Dataset(columns=data_columns, max_size=max_rows,data_where=data_where)
        print("  --> Train Data Loaded")
        print("  --> Loading Test Data")        
        if test_data_path is not None:
            self.test_data = Dataset(columns=None)
            self.test_data._data = load(test_data_path)
        else:
            self.test_data = Dataset(table_name="test",columns=data_columns, max_size=int(max_rows/5),data_where=data_where)
        print("  --> Loading Test Data")                

    def train(self,degree=2):
        if self.model.model is not None:
            print("Warning: Existing model will be overwritten.")
            response = ''
            while response != 'y' and response != 'n':
                response = input("Continue? (Y/n) > ").lower()
            if response == 'n':
                return
        self.train_results = self.model.train(self.target, self.train_data, degree=degree)

    def test(self):
        self.test_results = self.model.test(self.test_data)

    def results(self):
        print("    --> Train Results:")
        print(self.format_results(self.train_results))
        print("    --> Test Results:")
        print(self.format_results(self.test_results))
        
    def format_results(self,results):
        if results is None:
            return '        No Results.\n'
        return reduce(lambda s1,s2: s1+s2,["         *  {0}: {1}\n".format(kv[0],kv[1]) for kv in results.items()])
    
    @classmethod
    def generate_model_path(cls, id, ext='.joblib', dirpath=os.path.dirname(os.path.abspath(__file__))+'/models/'):
        return os.path.join(dirpath, id)

    @classmethod
    def load_model_from_id(cls,id,ext='.joblib',dir=None):
        if dir is None:
            path = cls.generate_model_path(id) 
        else:
            path = os.path.join(dir, id)
        if os.path.exists(path):
            pm = PolyReg(id,path=path)
            return pm
        else:
            return None
    
    @classmethod
    def load_model_from_path_s(cls,path):
        if os.path.exists(path):
            pm = PolyReg(id,path=path)
            return pm
        else:
            return None

    def load_model_from_path(self,path):
        if os.path.exists(path):
            print("Loading PolyReg...")
            self.train_data._data = load(os.path.join(path,'train_data.joblib'))
            print(" --> Train Data loaded.")
            self.test_data._data = load(os.path.join(path,'test_data.joblib'))
            print(" --> Test Data loaded.")
            self.model._model = load(os.path.join(path,'model.joblib'))
            print(" --> Model loaded.")
            if os.path.exists(os.path.join(path,'target.joblib')):
                self.target = load(os.path.join(path,'target.joblib'))
                self.model._target = [self.target]
                self.model._features = self.train_data.columns
                self.model._features.remove(self.target)
                print("   --> Model metadata loaded.")
            self.model._poly_features = load(os.path.join(path,'poly_features.joblib'))
            print(" --> PolynomialFeatures loaded.")
            self.model_path = path
            print('PolyReg loaded.')

        else:
            print('PolyReg files not found')


    def save(self):
        if self.model_path is None:
            self.save_as(self.model_path)
        if self.model_path is None: return
        try:
            os.mkdir(self.model_path)
        except FileExistsError as error:
            pass
        if self.target is not None:
            dump(self.target, os.path.join(self.model_path,'target.joblib'))
        dump(self.model._model, os.path.join(self.model_path,'model.joblib'))
        dump(self.model._poly_features, os.path.join(self.model_path,'poly_features.joblib'))
        dump(self.train_data._data, os.path.join(self.model_path,'train_data.joblib'))
        dump(self.test_data._data, os.path.join(self.model_path,'test_data.joblib'))
    
    def save_file(self,file,dir_path=None):
        if dir_path is None:
                dir_path = self.set_path()
                if dir_path is None: return
        try:
            os.mkdir(dir_path)
        except FileExistsError as error:
            pass
        if os.path.exists(os.path.join(dir_path,file+'.joblib')):
            os.remove(os.path.join(dir_path,file+'.joblib'))
        if file == 'model':
            if self.target is not None:
                dump(self.target, os.path.join(dir_path,'target.joblib'))
            dump(self.model._model, os.path.join(dir_path,'model.joblib'))
        elif file == 'train_data':
            dump(self.train_data._data, os.path.join(dir_path,'train_data.joblib'))
        elif file == 'test_data':
            dump(self.test_data._data, os.path.join(dir_path,'test_data.joblib'))

    def set_path(self):
        path = easygui.diropenbox(msg='Save as..',title='Save PolyReg Files',default="./")
        if path is None: return
        return path


    def save_as(self,path):
        print("Save as...")
        while True:
            if path is not None:
                self.model_path = path
                files_exist = False
                for file in ['model','poly_features','train_data','test_data']:
                    if os.path.exists(os.path.join(path,file+'.joblib')):
                        files_exist = True
                if not files_exist: 
                    self.save()
                    return
                print("Warning: Existing files will be overwritten.")
                response = ''
                while response != 'y' and response != 'n':
                    response = input("Continue? (Y/n) > ").lower()
                if response == 'y':
                    for file in ['model','poly_features','train_data','test_data']:
                        if os.path.exists(os.path.join(path,file+'.joblib')):
                            os.remove(os.path.join(path,file+'.joblib'))
            else:
                path = easygui.diropenbox(msg='Save as..',title='Save PolyReg Files',default="./")
                if path is None: return
                files_exist = False

if __name__ == '__main__':
    pm = PolyReg('full_model',max_rows=None)
    pm.train()
    pm.test()
    pm.results()
    pm.save_as(PolyReg.generate_model_path('full_model'))