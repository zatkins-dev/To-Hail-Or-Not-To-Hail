import pandas as pd
from google import auth
from google.cloud import bigquery,bigquery_storage_v1beta1
import time

class Dataset():
    def __init__(self, table_name='train',columns=None, max_size=None, data_where=None):
        self._data = None
        self._table_name = table_name
        self._data_where = None
        if columns is not None:
            self._columns = list(columns)
            if max_size is None:
                self.load_new_data()
            else:
                self.load_new_data_partial(max_rows=max_size)
        else:
            self._columns = None

    def __del__(self):
        del self._data

    def load_new_data_partial(self,max_rows):
        client = bigquery.Client()
        dtypes = dict([(c, 'float64') for c in self._columns])
        table_ref = bigquery.table.TableReference.from_string('to-hail-or-not-to-hail.gsod_copy.{}'.format(self._table_name))
        table = client.get_table(table_ref)
        schema_subset = [col for col in table.schema if col.name in self._columns]
        rows = client.list_rows(table, selected_fields=schema_subset,max_results=max_rows)
        print("    --> Generating dataframe...")
        start = time.perf_counter()
        self._data = rows.to_dataframe(dtypes=dtypes)
        cost = time.perf_counter() - start
        print("    --> Generated dataframe ({} s).".format(cost))
        if self._data_where is not None:
            self._data = self._data.loc[self._data.index.map(self._data_where),:].dropna()
        else:
            start = time.perf_counter()
            self._data = self._data.dropna()
            cost = time.perf_counter() - start
            print("    --> Removed n/a values ({} s).".format(cost))

    def load_new_data(self):
        storage_client = bigquery_storage_v1beta1.BigQueryStorageClient()
        
        client = bigquery.Client()

        table_reference = bigquery_storage_v1beta1.types.TableReference(
            project_id= 'to-hail-or-not-to-hail',
            dataset_id= 'gsod_copy',
            table_id= self._table_name,
        )
        parent = 'projects/to-hail-or-not-to-hail'
        read_options = bigquery_storage_v1beta1.types.TableReadOptions()
        read_options.selected_fields.extend(self._columns)
        session = storage_client.create_read_session(table_reference, parent, read_options=read_options)
        reader = storage_client.read_rows(
            bigquery_storage_v1beta1.types.StreamPosition(stream=session.streams[0])
        )
        dtypes = dict([(c, 'float64') for c in self._columns])
        print("  --> Generating dataframe...")
        start = time.perf_counter()
        self._data = reader.to_dataframe(session,dtypes=dtypes)
        cost = time.perf_counter() - start
        print("  --> Generated dataframe ({} s).".format(cost))
        if self._data_where is not None:
            self._data = self._data.loc[self._data.index.map(self._data_where),:].dropna()
        else:
            start = time.perf_counter()
            self._data = self._data.dropna()
            cost = time.perf_counter() - start
            print("  --> Removed n/a values ({} s).".format(cost))

    @property
    def columns(self):
        if self._columns is None and self._data is not None:
            self._columns = list(self._data.columns)
        return self._columns

    @property
    def data(self):
        return self._data
