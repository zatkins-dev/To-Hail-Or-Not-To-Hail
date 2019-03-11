import pandas as pd
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
import itertools
import time

class CalculateLinearModels:
    """The CalculateLinearModels class will employ exhaustive search to
    calculate the linear models of all combinations of predictors, then
    ranks the models by RSS.

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
            that is being predicted.
            
    Inspired from: http://www.science.smith.edu/~jcrouser/SDS293/labs/lab8-py.html"""
    
    def __init__(self, fileName, maxFeatures, numModels, outcomeName):
        """Initializes CalculateLinearModels, including reading in data.
        
        Raises:
            IOError: The data file has missing or invalid information.
            ValueError: The outcome name given is not a column in the
                data file."""
        self._maxFeatures = maxFeatures
        self._numModels = numModels
        self._nameOutcome = outcomeName

        self._listModels = []
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
            raise ValueError("The outcome given is not in the data file.")

        # With error checking complete, the dataframe needs to be split into
        # a feature dataframe and an outcome dataframe.
        self._outcomeDataframe = self._dataframe[self._nameOutcome]
        self._featureDataframe = self._dataframe.drop([self._nameOutcome], axis=1).astype('float64')

        # Generate all models. This will get expensive. The first for loop handles the number of
        # features, and the second for loop handles all the possible combinations of that number
        # of features.
        tic = time.time()
        for i in range(1, self._maxFeatures + 1):
            for combination in itertools.combinations(self._featureDataframe.columns.values.tolist(), i):
                self._listModels.append(self.generateModel(combination))
        toc = time.time()
        print("Time to generate models:", (toc-tic), "seconds.")

        self.mergeSort(self._listModels)
        
        for i in range(self._numModels):
            self._listBestModels.append(self._listModels[i])
        
        print("Model computation complete.")
    
    def generateModel(self, featureSet):
        """Creates a linear model with the feature set. 
        
        This allows for a user to call this function on specific features to
        get a specific linear model. 

        Args:
            featureSet: The set of features to build a linear model with.

        Returns:
            A list of a linear model and its RSS.
        
        Raises:
            ValueError: One or more features in the feature set do not exist.
        """

        # Ensure all of the features in the feature set exists in the feature 
        # data frame, raise exception otherwise.
        for feature in featureSet:
            if(not(feature in self._columnHeaders)):
                raise ValueError("One or more features in the feature set do not exist.")
        
        tempModel = sm.OLS(self._outcomeDataframe, self._featureDataframe[list(featureSet)]) # ordinary least squares
        regressionModel = tempModel.fit()
        RSS = ((regressionModel.predict(self._featureDataframe[list(featureSet)]) - self._outcomeDataframe) ** 2).sum()

        return [regressionModel, RSS]

    def mergeSort(self, list):
        """The joys of computer science. This merge sort is designed to sort the model list by RSS
        in ascending order. This will allow for the first n elements of the list to be the so called best models.

        Inspired by the Python3 implementation of: https://www.geeksforgeeks.org/merge-sort/ 
        """
        if len(list) > 1:
            list1 = list[:len(list)//2]
            list2 = list[len(list)//2:]

            # Activate the recursion!
            self.mergeSort(list1)
            self.mergeSort(list2)

            i = j = k = 0

            while(i < len(list1) or j < len(list2)):
                if(i < len(list1) and j < len(list2)):
                    if(list1[i][1] < list2[j][1]):
                        list[k] = list1[i]
                        i += 1
                    else:
                        list[k] = list2[j]
                        j += 1
                    k += 1
                elif i < len(list1):
                    list[k] = list1[i]
                    i += 1
                    k += 1
                else:
                    list[k] = list2[j]
                    j += 1
                    k += 1
    
    def summarizeModels(self):
        """
        A getter that prints out the regression model statistics for all of the "best" models.
        """
        for i in range(len(self._listBestModels)):
            print(self._listBestModels[i][0].summary())
            print("\n")

    def visualizeModels(self):
        """
        Create graphs of the regression models fit to the data.
        """

        