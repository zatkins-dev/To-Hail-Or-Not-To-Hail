from google.cloud import bigquery

import csv
import os.path

ALL_COLS = ("stn", "wban", "year", "mo", "da", "temp", "count_temp", "dewp", 
    "count_dewp", "slp", "count_slp", "stp", "cound_stp", "visib", "count_visib",
    "wdsp", "count_wdsp", "mxpst", "gust", "max", "flag_max", "min", "flag_min",
    "prcp", "flag_prcp", "sndp", "fog", "rain_drizzle", "snow_ice_pellets",
    "hail", "thunder", "tornado_funnel_cloud")

def get_full_table(year):
    client = bigquery.Client()
    gsod_dataset_ref = client.dataset('noaa_gsod', project='bigquery-public-data')
    gsod_dset = client.get_dataset(gsod_dataset_ref)
    gsod_full = client.get_table(gsod_dset.table(f'gsod{year}'))

    return  client.list_rows(gsod_full)
    
def is_bad_row(row, schema):
    bad_dict = {
        "temp": 9999.9,
        "dewp": 9999.9,
        "slp":  9999.9,
        "stp":  9999.9,
        "visib": 999.9,
        "wdsp": 999.9,
        "mxpsp": 999.9,
        "gust": 999.9,
        "max": 9999.9,
        "min": 9999.9,
        # "prcp": 99.99,  # prcp and sndp may need a deeper look than just this
        # "sndp": 999.9,
    }

    for col in schema:
        if col.name in bad_dict and row[col.name] == bad_dict[col.name]:
            return True

    return False

def clean_table(table):
    return [row for row in table if not is_bad_row(row, table.schema)]


def table_to_csv(year):
    # quick check to prevent duplicate requests
    if os.path.exists(f'csv/gsod{year}.csv'):
        print("File for this year already exists")
        return False

    # first, get the data
    print(f'Getting data for {year}')
    full_table = get_full_table(year)
    print("Table found, now cleaning")
    cleaned_table = clean_table(full_table)

    # check if any clean data was actually found
    if len(cleaned_table) == 0:
        # if none was found, jump ship
        print(f'No clean data was found for {year}. Exiting...')
        return False
    else:
        print(f"{len(cleaned_table)} rows retrieved! Writing to file...")

    # create a csv file, and a writer to fill it
    csv_file = open(f'csv/gsod{year}.csv', 'w')
    csv_writer = csv.writer(csv_file)

    # fill the table using the data retrieved
    csv_writer.writerows(cleaned_table)

    print("Done!")

    # close the file
    csv_file.close()

    return True
