import pandas as pd
from google.cloud import bigquery

class Dataset():
    def __init__(self, table_name='train',columns=None, max_size=None, data_where=None):
        self._max_size = max_size
        self._columns = list(columns)
        self._data = None
        self._table_name = table_name
        self._data_where = None
        self.load_new_data()

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
        if self._columns.count('mo')>0:
            self._data = self._data.astype({'mo':'float64'})
        if self._columns.count('prcp')>0:
            self._data = self._data.astype({'prcp':'float64'})
        if self._data_where is not None:
            self._data = self._data.loc[self._data.index.map(self._data_where),self._columns].dropna()
        else:
            self._data = self._data.loc[:, self._columns].dropna()
        self._data = self._data

    @property
    def columns(self):
        return self._columns

    @property
    def data(self):
        return self._data
