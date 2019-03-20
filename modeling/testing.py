from google.cloud import bigquery
from functools import reduce

#Get the test set
def get_test_set(columns=("stn", "hail"),max_rows=100000000):
    client = bigquery.Client()

    gsod_project_test = client.dataset('gsod_copy', project='to-hail-or-not-to-hail')
    gsod_test_full = client.get_table('test')

    schema_subset = [col for col in gsod__test_full.schema if col.name in columns]
    return  [x for x in client.list_rows(gsod_full, start_index=0, selected_fields=schema_subset, max_results=min((gsod_full.num_rows,max_rows)))


def test_regression(test_name="Some Algorithm"):
    #Variables for the test
    results = get_cleaned_table()
    error_min_dif = 1000.0
    error_max_dif = 0.0
    error_sum_dif = 0.0
    set_size = len(results)

    for row in results:
        estimated_temperature = algoithm(row)
        temperature_error = expected_temperature - row.temp

        if temperature_error > error_min_dif:
            error_min_dif = temperature_error

        if temperature_error > error_max_dif:
            error_max_dif = temperature_error

        error_sum_dif += temperature_error

    print(test_name)
    print(f"Test set size: {set_size}")
    print(f"Minimum error: {error_min_dif}")
    print(f"Maximum error: {error_max_dif}")
    print(f"Average error: ({error_sum_dif} / {set_size})")
