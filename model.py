# -*- coding: utf-8 -*-
"""
Created on Tue Sep  1 14:00:29 2020

@author: Dennis
"""
import pandas as pd
import numpy as np

from sklearn.linear_model import LinearRegression

import pickle
from pickle import dump, load


df = pd.read_csv("retailnew.csv")

df = df[['DaySum','MonthAvg','MonthSum']]

X = df.iloc[:,0:2]
y = df.iloc[:,2]

lr = LinearRegression()

lr.fit(X,y)

#Save Model

filename = 'model.sav'

dump(lr,open(filename,'wb'))

print("Model Saved")


# Saving the data columns from training

filename = 'model_columns.sav'

model_columns = list(X.columns)
dump(model_columns, open(filename,'wb'))

print("Models columns dumped!")
