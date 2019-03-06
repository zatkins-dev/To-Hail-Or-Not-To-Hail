from google.cloud import bigquery

client = bigquery.Client()

query = client.query("""
    SELECT *
    FROM `bigquery-public-data.noaa_gsod.stations`
    WHERE stn=30050 AND year=1929 AND mo=10
""")

results=query.results()

print("stn","year","mo","da","temp", "dewp", "slp", "hail")
for row in results:
    print(row.stn, row.year, row.mo, row.da, row.temp, row.dewp, row.slp, row.hail)