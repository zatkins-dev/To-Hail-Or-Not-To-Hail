
from google.cloud import bigquery

client = bigquery.Client()

gsod_dataset_ref = client.dataset('noaa_gsod', project='bigquery-public-data')

gsod_dset = client.get_dataset(gsod_dataset_ref)

gsod_full = client.get_table(gsod_dset.table('gsod2015'))

schema = ""

for scheme in gsod_full.schema:
    schema += (f"{scheme.name}:{scheme.field_type},")

print (schema)