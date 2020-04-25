import numpy as np
import math
import matplotlib.pyplot as plt

'''
Machine learning NaiveBayes classifier
'''
class NaiveBayesClassifier:

    def __init__(self):
        self.feature = np.array()
        self.output = np.array()
    
    '''
    method to train model
    '''
    def fit(self, trainX, trainY):
        pass
    
    '''
    method that predicts class for given feature set
    '''
    def predict(self, testX):
        predictedClass = np.array()
        return predictedClass
    
    '''
    method to calculate model's accuracy
    '''
    def getAccuracy(self, actualClass, predictedClass):
        pass