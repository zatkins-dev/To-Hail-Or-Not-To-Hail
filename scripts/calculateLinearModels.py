import pandas as pd
import numpy as np
import statsmodels.api
import matplotlib.pyplot as plt

class CalculateLinearModels:
    """The CalculateLinearModels class will employ best subset
    selection to generate the 'best' linear regression models.

    The CalculateLinearModels class will first read in the data stored
    in the fileName, then ensure all data is clean before continuing.
    Clean in this context refers to a data set with no missing values
    and all numeric values.

    Once the data passes all checks, a linear regression model will
    be fit to all possible combinations of features in the data, 
    up to a specified number, and an RSS will be calculated for all 
    linear regression models.

    Then, the class will store a list of a specified length of linear
    regression models.

    The regression models and their visualizations can be returned or
    viewed with their specified functions. 

    Arguments:
        fileName: A string which has the name of the data file that
            needs to be read in.
        maxFeatures: An integer that sets the maximum number of 
            features in each linear model.
        numModels: An integer that sets the number of linear models to
            store. If the number of linear models that can be generated
            are smaller than numModels, the class will have a list of
            all the models.
        outcomeName: A string of the name of the column in the dataset 
            that is being predicted."""
    
    def __init__(self, fileName, maxFeatures, numModels, outcomeName):
        """Initializes CalculateLinearModels, including reading in data.
        
        Raises:
            IOError: The data file has missing or invalid information.
            NameError: The outcome name given is not a column in the
                data file."""
        self._maxFeatures = maxFeatures
        self._numModels = numModels
        self._nameOutcome = outcomeName

        self._listModels = []
        self._listModelsRSS = []
        self._listBestModels = []

        self._dataframe = pd.read_csv(fileName)
        self._columnHeaders = self._dataframe.columns.values.tolist()

        # Ensure that the data frame holds no non-numeric values,
        # raise exception otherwise.
        numericFeatures = self._dataframe.applymap(np.isreal).all()
        for numericFeature in numericFeatures:
            if(not numericFeature):
                raise IOError("The datafile has missing or invalid information.")

        # Ensure the outcome name is one of the column names,
        # raise exception otherwise.
        if(not(self._nameOutcome in self._columnHeaders)):
            raise NameError("The outcome given is not in the data file.")

        # With error checking complete, the dataframe needs to be split into
        # a feature dataframe and an outcome dataframe.
        self._outcomeDataframe = self._dataframe[self._nameOutcome]
        self._featureDataframe = self._dataframe.drop([self._nameOutcome], axis=1).astype('float64')

        # testing
        print(self._featureDataframe.head())
        print(self._outcomeDataframe.head())
