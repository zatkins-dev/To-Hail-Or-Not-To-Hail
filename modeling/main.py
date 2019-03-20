from . import PolyReg
from functools import reduce

def main():
    cols = ['mo','temp', 'dewp','slp', 'stp', 'visib', 'wdsp', 'altitude', 'latitude', 'prcp','fog','rain_drizzle','snow_ice_pellets','hail','tornado_funnel_cloud']
    target = 'temp'
    model_choice = 0
    while not (model_choice == 1 or model_choice == 2):
        try:
            model_choice = int(input("Create a test model (1) or full model (2)? > "))
        except:
            model_choice = 0
    
    id = None
    degree = 0
    id = str(input("Model id: > "))
    degree = int(input("Degree of polynomial regression [2]: > ") or 2)
    train_data_path = input("Train data path [None]: > ") or None
    test_data_path = input("Test data path [None]: > ") or None
    if model_choice == 1:
        numrows = int(input("Max number of rows in model [100000]: > ") or 100000)
        print("Creating test model.")
        model = PolyReg(id,max_rows=numrows,train_data_path=train_data_path,test_data_path=test_data_path,data_columns=cols)
    elif model_choice == 2:
        print("Creating full model.")
        model = PolyReg(id,max_rows=None,train_data_path=train_data_path,test_data_path=test_data_path,data_columns=cols)
    
    print("  --> Data Loaded")
    model.save_file('train_data',dir_path=PolyReg.generate_model_path(id))
    model.save_file('test_data',dir_path=PolyReg.generate_model_path(id))

    train_desc = model.train_data._data.describe()
    test_desc = model.test_data._data.describe()
    print("    --> Train Data Summary")
    print(train_desc.to_string())
    print("    --> Test Data Summary")
    print(test_desc.to_string())
    print("  --> Model Initialized")
    model.train(degree=degree)
    print("  --> Model Trained")
    features = cols
    features.remove(target) 
    if degree > 1:
        features = model.model._poly_features.get_feature_names(features)
    coefs = model.model.model.coef_[0]
    inter = model.model.model.intercept_[0]
    eqn = "{}".format(round(inter,5))
    for i in range(1,len(features)):
        if len(eqn)>64 and eqn.rfind('\n',len(eqn)-65) == -1:
            eqn += "\n            "+" + "+"{}({})".format(round(coefs[i],5),features[i])
        else:
            eqn += " + "+"{}({})".format(round(coefs[i],5),features[i])
    print("    --> Model equation:")
    print("        {} = {}".format(target,eqn))
    model.test()
    print("  --> Model Tested")
    model.results()
    model.save_file('model',dir_path=PolyReg.generate_model_path(id))

if __name__ == '__main__':
    main()
