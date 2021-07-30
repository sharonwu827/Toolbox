

# cross validation to evaluate the model performance
# import package

import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold

def cross_validation(X,y,k,shuffle = True):
    # prepare the cross-validation procedure
    cv = KFold(n_splits=k, random_state=1, shuffle=shuffle)
    # create the model
    linear_reg = LinearRegression()
    # evaluate the model
    scores = cross_val_score(linear_reg, X, y
                             , cv=cv
                             , n_jobs=-1 # using all processors
                             ,scoring='accuracy')
    # report performance
    print('Accuracy: %.3f (%.3f)' % (mean(scores), std(scores)))






