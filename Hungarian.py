# -*- coding: utf-8 -*-
"""
Created on Wed May 31 19:22:30 2023

@author: avtei
"""
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas
from scipy.optimize import linear_sum_assignment
#https://docs.scipy.org/doc/scipy-0.18.1/reference/generated/scipy.optimize.linear_sum_assignment.html
#The student that didnt make 10 choices gets his first choice as 2 instead of 1 for doing it wrong

pandas.set_option('display.max_columns', 10)

ChoiceMin = 1
ChoiceMax = 88

projects = choices = pandas.read_csv("Project List by Section.csv", index_col=0)
choices = pandas.read_csv("Project-Data.csv", index_col=0)
choices.columns = [int(x) for x in choices.columns]
Result = pandas.DataFrame(index=choices.index, columns=["Project"])
print(choices)

CostMatrix = pandas.DataFrame(index=choices.index, columns=np.arange(1,88))

#its a cost matrix so we want to price the choices low and the projects not chosen very highly
HighCost = 100
CostMatrix[:] = HighCost

for student in choices.index:
    #Cost penality for invalid choices
    CostPenalty = np.where((choices.loc[student] < ChoiceMin) | (choices.loc[student] > ChoiceMax))[0].shape[0]
    ValidChoices = np.where((choices.loc[student] > ChoiceMin) & (choices.loc[student] < ChoiceMax))[0]
    for cost,choice in enumerate(choices.loc[student].iloc[ValidChoices]):
        CostMatrix.at[student, choice] = cost + CostPenalty
print(CostMatrix)


# =============================================================================
# # Set up the matplotlib figure
# f, ax = plt.subplots(figsize=(11, 9))
# # Generate a custom diverging colormap
# cmap = sns.diverging_palette(230, 20, as_cmap=True)
# # Draw the heatmap with the mask and correct aspect ratio
# sns.heatmap(CostMatrix.astype(np.float64), cmap=cmap, vmax=HighCost, vmin=0, center=0, square=True, linewidths=.5, cbar_kws={"shrink": .5})
# =============================================================================

row_ind, col_ind = linear_sum_assignment(CostMatrix)

Result = pandas.DataFrame(index=choices.index, columns=["Project", "Choice #"])

for i in range(len(row_ind)):
    student = CostMatrix.index[row_ind[i]]
    project = CostMatrix.columns[col_ind[i]]
    Result.at[student, "Project"] = project
    try:
        Result.at[student, "Choice #"] = np.where(choices.loc[student] == project)[0][0]+1
    except:
        Result.at[student, "Choice #"] = -1
    
    
print(Result)

print("="*50)
print("Result summary:")
print("Average choice that each student got:", Result[Result["Choice #"] != -1]["Choice #"].values.astype(np.float64).mean())
print("Did any student get assigned a project that wasnt on their list of choices:", (Result["Choice #"] == -1).any())

Result.to_csv("Project-assignment.csv")