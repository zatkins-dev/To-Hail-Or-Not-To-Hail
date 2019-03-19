from . import PolyReg

def main():
    model_choice = 0
    while not (model_choice == 1 or model_choice == 2):
        try:
            model_choice = int(input("Create a test model (1) or full model (2)? > "))
        except:
            model_choice = 0
    
    id = None
    degree = 0
    if model_choice == 1:
        id = str(input("Model id: > "))
        numrows = int(input("Max number of rows in model: > "))
        degree = int(input("Degree of polynomial regression: > "))
        print("Creating test model.")
        model = PolyReg(id,max_rows=numrows,data_columns=['mo','temp', 'dewp','slp', 'stp', 'visib', 'wdsp', 'altitude', 'latitude', 'prcp','fog','rain_drizzle','snow_ice_pellets','hail','tornado_funnel_cloud'])
    elif model_choice == 2:
        id = str(input("Model id: > "))
        degree = int(input("Degree of polynomial regression: > "))
        print("Creating full model.")
        model = PolyReg(id,max_rows=None,data_columns=['mo','temp', 'dewp','slp', 'stp', 'visib', 'wdsp', 'altitude', 'latitude', 'prcp','fog','rain_drizzle','snow_ice_pellets','hail','tornado_funnel_cloud'])
    
    print("  --> Data Loaded")
    train_desc = model.train_data._data.describe()
    test_desc = model.test_data._data.describe()
    print("    --> Train Data Summary")
    print(train_desc.to_string())
    print("    --> Test Data Summary")
    print(test_desc.to_string())
    print("  --> Model Initialized")
    model.train(degree=degree)
    print("  --> Model Trained")
    model.test()
    print("  --> Model Tested")
    model.results()
    model.save_as(PolyReg.generate_model_path(id))


if __name__ == '__main__':
    main()
