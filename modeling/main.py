from . import PolyReg
from functools import reduce
import os

deg2features = ['1', 'mo', 'dewp', 'slp', 'stp', 'visib', 'wdsp', 'altitude', 'latitude', 'prcp', 'fog', 'rain_drizzle', 'snow_ice_pellets', 'hail', 'tornado_funnel_cloud', 'mo^2', 'mo dewp', 'mo slp', 'mo stp', 'mo visib', 'mo wdsp', 'mo altitude', 'mo latitude', 'mo prcp', 'mo fog', 'mo rain_drizzle', 'mo snow_ice_pellets', 'mo hail', 'mo tornado_funnel_cloud', 'dewp^2', 'dewp slp', 'dewp stp', 'dewp visib', 'dewp wdsp', 'dewp altitude', 'dewp latitude', 'dewp prcp', 'dewp fog', 'dewp rain_drizzle', 'dewp snow_ice_pellets', 'dewp hail', 'dewp tornado_funnel_cloud', 'slp^2', 'slp stp', 'slp visib', 'slp wdsp', 'slp altitude', 'slp latitude', 'slp prcp', 'slp fog', 'slp rain_drizzle', 'slp snow_ice_pellets', 'slp hail', 'slp tornado_funnel_cloud', 'stp^2', 'stp visib', 'stp wdsp', 'stp altitude', 'stp latitude', 'stp prcp', 'stp fog', 'stp rain_drizzle', 'stp snow_ice_pellets', 'stp hail', 'stp tornado_funnel_cloud', 'visib^2', 'visib wdsp', 'visib altitude', 'visib latitude', 'visib prcp', 'visib fog', 'visib rain_drizzle', 'visib snow_ice_pellets', 'visib hail', 'visib tornado_funnel_cloud', 'wdsp^2', 'wdsp altitude', 'wdsp latitude', 'wdsp prcp', 'wdsp fog', 'wdsp rain_drizzle', 'wdsp snow_ice_pellets', 'wdsp hail', 'wdsp tornado_funnel_cloud', 'altitude^2', 'altitude latitude', 'altitude prcp', 'altitude fog', 'altitude rain_drizzle', 'altitude snow_ice_pellets', 'altitude hail', 'altitude tornado_funnel_cloud', 'latitude^2', 'latitude prcp', 'latitude fog', 'latitude rain_drizzle', 'latitude snow_ice_pellets', 'latitude hail', 'latitude tornado_funnel_cloud', 'prcp^2', 'prcp fog', 'prcp rain_drizzle', 'prcp snow_ice_pellets', 'prcp hail', 'prcp tornado_funnel_cloud', 'fog^2', 'fog rain_drizzle', 'fog snow_ice_pellets', 'fog hail', 'fog tornado_funnel_cloud', 'rain_drizzle^2', 'rain_drizzle snow_ice_pellets', 'rain_drizzle hail', 'rain_drizzle tornado_funnel_cloud', 'snow_ice_pellets^2', 'snow_ice_pellets hail', 'snow_ice_pellets tornado_funnel_cloud', 'hail^2', 'hail tornado_funnel_cloud', 'tornado_funnel_cloud^2']

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
    output = None
    degree = 0
    id = str(input("Model id: > "))
    if not os.path.exists(PolyReg.generate_model_path(id)):
        os.mkdir(PolyReg.generate_model_path(id))
    output_path = os.path.join(PolyReg.generate_model_path(id),'output.txt')
    degree = int(input("Degree of polynomial regression [2]: > ") or 2)
    train_data_path = input("Train data path [None]: > ") or None
    test_data_path = input("Test data path [None]: > ") or None
    if model_choice == 1:
        numrows = int(input("Max number of rows in model [100000]: > ") or 100000)
        if os.path.exists(output_path):
            os.remove(output_path)
        output = open(output_path,'w')
        print("Opening output file at {}".format(output_path))
        print("Creating test model.")
        output.write("Creating test model.\n")
        model = PolyReg(id,max_rows=numrows,train_data_path=train_data_path,test_data_path=test_data_path,data_columns=cols)
    elif model_choice == 2:
        output = open(output_path,'w')
        print("Creating full model.")
        print("Opening output file at {}".format(output_path))
        output.write("Creating full model.\n")
        model = PolyReg(id,max_rows=None,train_data_path=train_data_path,test_data_path=test_data_path,data_columns=cols)
    
    print("  --> Data Loaded")
    output.write("  --> Data Loaded\n")
    if train_data_path is not None and train_data_path!=os.path.join(PolyReg.generate_model_path(id),'train_data.joblib'):
        model.save_file('train_data',dir_path=PolyReg.generate_model_path(id))
    if test_data_path is not None and train_data_path!=os.path.join(PolyReg.generate_model_path(id),'test_data.joblib'):
        model.save_file('test_data',dir_path=PolyReg.generate_model_path(id))

    train_desc = model.train_data._data.describe()
    test_desc = model.test_data._data.describe()
    print("    --> Train Data Summary")
    output.write("  --> Train Data Summary\n")
    print(train_desc.to_string())
    output.write(train_desc.to_string())
    output.write("\n")
    print("    --> Test Data Summary")
    output.write("    --> Test Data Summary\n")
    print(test_desc.to_string())
    output.write(test_desc.to_string())
    output.write("\n")
    print("  --> Model Initialized")
    output.write("  --> Model Initialized\n")
    model.train(degree=degree)
    print("  --> Model Trained")
    output.write("  --> Model Trained\n")
    features = cols
    features.remove(target) 
    if degree > 1:
        try:
            features = model.model._poly_features.get_feature_names(features)
        except:
            features = deg2features
    coefs = model.model.model.coef_[0]
    inter = model.model.model.intercept_[0]
    eqn = "{}".format(round(inter,5))
    for i in range(1,len(features)):
        if len(eqn)>64 and eqn.rfind('\n',len(eqn)-65) == -1:
            eqn += "\n            "+" + "+"{}({})".format(round(coefs[i],5),features[i])
        else:
            eqn += " + "+"{}({})".format(round(coefs[i],5),features[i])
    print("    --> Model equation:")
    output.write("    --> Model equation:\n")
    print("        {} = {}".format(target,eqn))
    output.write("        {} = {}\n".format(target,eqn))
    model.test()
    print("  --> Model Tested")
    output.write("  --> Model Tested\n")
    model.results()
    model.results(write=output.write)
    output.close()
    model.save_file('model',dir_path=PolyReg.generate_model_path(id))

if __name__ == '__main__':
    main()
