
from google.cloud import bigquery

client = bigquery.Client()

gsod_dataset_ref = client.dataset('gsod_copy', project='to-hail-or-not-to-hail')

gsod_dset = client.get_dataset(gsod_dataset_ref)

gsod_full = client.get_table(gsod_dset.table('data'))

schema = ""

for scheme in gsod_full.schema:
    schema += (f"{scheme.name},")# {scheme.field_type},")

schema = schema.replace("FLOAT", "FLOAT64")
schema = schema.replace("INTEGER", "INT64")



print (schema)