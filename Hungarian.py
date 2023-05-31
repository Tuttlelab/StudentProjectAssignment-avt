# -*- coding: utf-8 -*-
"""
Created on Wed May 31 19:22:30 2023

@author: avtei
"""
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas, sys, copy
from scipy.optimize import linear_sum_assignment
#https://docs.scipy.org/doc/scipy-0.18.1/reference/generated/scipy.optimize.linear_sum_assignment.html
#The student that didnt make 10 choices gets his first choice as 2 instead of 1 for doing it wrong

pandas.set_option('display.max_columns', 10)


projects = pandas.read_csv("Project List by Section.csv", index_col=0)
assert (projects["Capacity"].values < 10).all()

ChoiceMin = min(projects.index)
ChoiceMax = max(projects.index)

# We allow for projects with capacity > 1 by making project 87 into projects 87 & 87.1
for i in projects.index:
    if projects.at[i, "Capacity"] > 1:
        for addition in range(1, projects.at[i, "Capacity"].astype(np.int64) +1):
            projects.at[i+(addition/10), "Capacity"] = 1    
            projects.at[i+(addition/10), "Section"] = projects.at[i, "Section"]    
        projects.at[i, "Capacity"] = 1

projects = projects.sort_index()
print(projects.iloc[-5:])



choices = pandas.read_csv("Project-Data.csv", index_col=0)
choices.columns = [int(x) for x in choices.columns]
Result = pandas.DataFrame(index=choices.index, columns=["Project"])
print(choices)

#Adjust the choices to account for projects with a capacity > 1
DecimalChoices = copy.copy(choices)
for student in choices.index:
    new_choice_list = []
    for choice in choices.loc[student].values:
        new_choice_list.append(choice)
        for decimal in np.arange(0.1, 1.0, 0.1):
            if float(choice)+decimal in projects.index:
                new_choice_list.append(float(choice)+decimal)
            else:
                break
    print(new_choice_list)
    for i in range(10):
        DecimalChoices.at[student, i+1] = new_choice_list[i]
print(DecimalChoices)


CostMatrix = pandas.DataFrame(index=DecimalChoices.index, columns=projects.index)
#its a cost matrix so we want to price the choices low and the projects not chosen very highly
HighCost = 10000
CostMatrix[:] = HighCost

for student in DecimalChoices.index:
    #Cost penality for invalid DecimalChoices
    CostPenalty = np.where((DecimalChoices.loc[student] < ChoiceMin) | (DecimalChoices.loc[student] > ChoiceMax))[0].shape[0]
    ValidChoices = np.where((DecimalChoices.loc[student] > ChoiceMin) & (DecimalChoices.loc[student] < ChoiceMax))[0]
    for cost,choice in enumerate(DecimalChoices.loc[student].iloc[ValidChoices]):
        CostMatrix.at[student, choice] = (cost**2) + CostPenalty
        #CostMatrix.at[student, choice] = cost + CostPenalty
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
    project = np.floor(project).astype(np.int64)
    Result.at[student, "Project"] = project
    try:
        Result.at[student, "Choice #"] = np.where(choices.loc[student] == project)[0][0]+1
    except:
        Result.at[student, "Choice #"] = -1
    
    
print(Result)

print("="*50)
print("Result summary:")
print("Number of students that got their first choice:", (Result["Choice #"] == 1).sum())
print("Number of students that got their second choice:", (Result["Choice #"] == 2).sum())
print("Number of students that got their third choice:", (Result["Choice #"] == 3).sum())

print("Average choice that each student got:", Result[Result["Choice #"] != -1]["Choice #"].values.astype(np.float64).mean())
print("Did any student get assigned a project that wasnt on their list of choices:", (Result["Choice #"] == -1).any())

Result.to_csv("Project-assignment.csv")