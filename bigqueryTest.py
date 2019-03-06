from google.cloud import bigquery
from functools import reduce

client = bigquery.Client()

gsod_dataset_ref = client.dataset('noaa_gsod', project='bigquery-public-data')

gsod_dset = client.get_dataset(gsod_dataset_ref)

gsod_full = client.get_table(gsod_dset.table('gsod1974'))

schema_subset = [col for col in gsod_full.schema if col.name in ("stn","year","mo","da","temp", "dewp", "slp", "stp", "wdsp", "hail")]
results = [x for x in client.list_rows(gsod_full, start_index=0, selected_fields=schema_subset, max_results=1000) 
                   if x.temp != 9999.9 and x.dewp != 9999.9 and x.slp != 9999.9 and x.stp != 9999.9 and x.wdsp != 999.9]

num_results = len(results)
num_hail = reduce(lambda acc,x2: acc + int(x2.hail), results,0)
num_nohail = len(results) - num_hail

print(f"Entries (No Hail/Hail): {num_results} ({num_nohail}/{num_hail})")
print("stn","year","mo","da","temp", "dewp", "slp", "stp", "wdsp", "hail")
for row in results:
    print(row.stn, row.year, row.mo, row.da, row.temp, row.dewp, row.slp, row.stp, row.wdsp, row.hail)