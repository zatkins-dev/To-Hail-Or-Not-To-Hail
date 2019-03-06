from google.cloud import bigquery

client = bigquery.Client()

gsod_dataset_ref = client.dataset('noaa_gsod', project='bigquery-public-data')

gsod_dset = client.get_dataset(gsod_dataset_ref)

gsod_full = client.get_table(gsod_dset.table('gsod1929'))

schema_subset = [col for col in gsod_full.schema if col.name in ("stn","year","mo","da","temp", "dewp", "slp", "hail")]
results = [x for x in client.list_rows(gsod_full, start_index=100, selected_fields=schema_subset, max_results=10)]

print("stn","year","mo","da","temp", "dewp", "slp", "hail")
for row in results:
    print(row.stn, row.year, row.mo, row.da, row.temp, row.dewp, row.slp, row.hail)