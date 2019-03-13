
# from google.cloud import bigquery

# client = bigquery.Client()

# gsod_dataset_ref = client.dataset('noaa_gsod', project='bigquery-public-data')

# gsod_dset = client.get_dataset(gsod_dataset_ref)

# gsod_full = client.get_table(gsod_dset.table('gsod2015'))

# schema = ""

# for scheme in gsod_full.schema:
#     schema += (f"{scheme.name}:{scheme.field_type},")

# print (schema)

from scripts import CalculateLinearModels

def main():
    choice = 0
    print("Welcome to the linear model test program!")
    menu()
    choice = input("Input: ")
    choice = int(choice)
    print("")
    while(choice != 4):
        if(choice == 1):
            fileName = input("Enter relative data file path: ")
            maxPredictors = input("Enter maximum number of parameters allowed in model: ")
            numModels = input("Enter number of models to be kept: ")
            outcomeName = input("Enter outcome column name: ")
            maxPredictors = int(maxPredictors)
            numModels = int(numModels)
            print("")
            models = CalculateLinearModels(fileName, maxPredictors, numModels, outcomeName)
        elif(choice == 2):
            models.summarizeModels()
        elif(choice == 3):
            models.visualizeModels()
        elif(choice != 4):
            print("Invalid choice!")

        print("")
        menu()
        choice = input("Input: ")
        choice = int(choice)
        print("")

def menu():
    print("Please make a choice.")
    print("1. Create new set of linear models.")
    print("2. See summary of linear models.")
    print("3. See visualization of linear models.")
    print("4. Exit.")

if __name__ == '__main__': main()
