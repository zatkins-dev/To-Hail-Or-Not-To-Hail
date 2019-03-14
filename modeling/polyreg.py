import numpy as np
import pandas as pd
from joblib import dump, load
from functools import reduce
import easygui 
import os
from . import Dataset,Model

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
