

query = ""
for year in range(1929,2018):
    query += (f"SELECT * FROM \n\t`bigquery-public-data.noaa_gsod.gsod{year}`\nUNION ALL\n")

print (query)