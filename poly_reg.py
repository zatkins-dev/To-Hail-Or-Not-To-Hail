import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from google.cloud import bigquery
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import PolynomialFeatures
from joblib import Parallel, dump, load
import os

class PolyReg():
    filename_model = ''
    compress_model = 0
    filename_data = ''
    compress_data = 0
    data = None
    model = None

    def __init__(self,fn_model='model_poly.joblib',model_compress=0,model_ext='',fn_data='data_poly.joblib.gz',data_compression=4):
        self.filename_data = fn_data
        if os.path.exists(self.filename_data):
            self.data = load(self.filename_data)
        else:
            self.load_data()
        self.filename_model = fn_model
        if os.path.exists(self.filename_model):
            self.model = load(self.filename_model)
        else:
            self.train_model()

    def __del__(self):
        dump(self.data,self.filename_data,compress=self.compress_data)
        dump(self.model,self.filename_model,compress=self.compress_model)


    def load_data(self):
        client = bigquery.Client()

        gsod_dataset_ref = client.dataset('gsod_copy', project='to-hail-or-not-to-hail')
        gsod_dset = client.get_dataset(gsod_dataset_ref)
        gsod_clean = client.get_table(gsod_dset.table('gsod_clean'))

        self.data = client.list_rows(gsod_clean,max_results=100000).to_dataframe()
        self.data = self.data.replace({'prcp':99.99}, 0.0)
        self.data = self.data.loc[:,['mo','temp','dewp','slp','stp','visib','wdsp','max','min','prcp']]

        dump(self.data,self.filename_data,compress=self.compress_data)

    def train_model(self):
        features = ['mo','temp','dewp','slp','stp','visib','wdsp','max','min']
        predict = ['prcp']
        self.model = PolynomialFeatures(degree = 4) 
        X_poly = self.model.fit_transform(self.data.loc[:,features])
        self.model.fit(X_poly,self.data.loc[:,predict])

