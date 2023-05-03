# -*- coding: utf-8 -*-
"""
Created on Wed May  3 14:15:11 2023

@author: rkb19187
"""
import numpy as np
import pandas

choices = pandas.read_csv("Project-Data.csv", index_col=0)
choices.columns = [int(x) for x in choices.columns]
Result = pandas.DataFrame(columns=["Project"])
print(choices)

#Check that it is logical
Result["Project"] = choices[1]

def valid():
    return np.unique(Result["Project"].values).shape[0] == Result["Project"].values.shape[0]

print(valid())

#Calculate a score
