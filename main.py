
from google.cloud import bigquery

client = bigquery.Client()

gsod_dataset_ref = client.dataset('gsod1929', project='bigquery-public-data')
