# -*- coding: utf-8 -*-
"""
Created on Wed May 31 19:22:30 2023

@author: avtei
"""
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas, sys, copy, os
from scipy.optimize import linear_sum_assignment
#https://docs.scipy.org/doc/scipy-0.18.1/reference/generated/scipy.optimize.linear_sum_assignment.html
#The student that didnt make 10 choices gets his first choice as 2 instead of 1 for doing it wrong

pandas.set_option('display.max_columns', 10)


os.chdir("Real")


# =============================================================================
# ADDITIONAL REQUIREMENTS
# •	Aim to give at least one student to each supervisor
# •	Make sure no staff capacities are breached
# =============================================================================

class StudentProjectAssignment:
    @property
    def ChoiceMin(self):
        return min(self.projects.index)
    @property
    def ChoiceMax(self):
        return max(self.projects.index)
    
    def load_projects(self):
        self.projects = pandas.read_csv("Project List by Section.csv", index_col=0)
        assert (self.projects["Capacity"].values < 10).all() # required, because of the project 68.0, 68.1 hack for 68 with 2 slots. WIll only work up to a capacity of 10
        # We allow for projects with capacity > 1 by making project 87 into projects 87 & 87.1
        for i in self.projects.index:
            if self.projects.at[i, "Capacity"] > 1:
                for addition in range(1, self.projects.at[i, "Capacity"].astype(np.int64)):
                    self.projects.at[i+(addition/10), "Capacity"] = 1    
                    self.projects.at[i+(addition/10), "Section"] = self.projects.at[i, "Section"]    
                self.projects.at[i, "Capacity"] = 1
        
        self.projects = self.projects.sort_index()
        print(self.projects.iloc[-5:])
        
    def load_choices(self):
        self.choices = pandas.read_csv("Project-Data.csv", index_col=0)
        self.choices.columns = [int(x) for x in self.choices.columns]
        print("\nchoices")
        print(self.choices)
        
        #Adjust the choices to account for projects with a capacity > 1
        self.DecimalChoices = copy.copy(self.choices).astype(float)
        for student in self.choices.index:
            new_choice_list = []
            for choice in self.choices.loc[student].values:
                new_choice_list.append(choice)
                for decimal in np.arange(0.1, 1.0, 0.1):
                    if float(choice)+decimal in self.projects.index:
                        new_choice_list.append(float(choice)+decimal)
                    else:
                        break
            #print(new_choice_list)
            for i in range(10):
                self.DecimalChoices.at[student, i+1] = new_choice_list[i]
        print("\nDecimalChoices")
        print(self.DecimalChoices)
    
    def calc_CostMatrix(self, adjustments=None):
        #adjustments = [{"Baum": 0.1}]
        self.CostMatrix = pandas.DataFrame(index=self.DecimalChoices.index, columns=self.projects.index)
        #its a cost matrix so we want to price the choices low and the projects not chosen very highly
        HighCost = 10000
        self.CostMatrix[:] = HighCost
        
        for student in self.DecimalChoices.index:
            #Cost penality for invalid DecimalChoices
            CostPenalty = np.where((self.DecimalChoices.loc[student] < self.ChoiceMin) | (self.DecimalChoices.loc[student] > self.ChoiceMax))[0].shape[0]
            ValidChoices = np.where((self.DecimalChoices.loc[student] >= self.ChoiceMin) & (self.DecimalChoices.loc[student] <= self.ChoiceMax))[0]
            for cost,choice in enumerate(self.DecimalChoices.loc[student].iloc[ValidChoices]):
                self.CostMatrix.at[student, choice] = (cost**2) + CostPenalty
                
                
# =============================================================================
#                 if adjustments is not None:
#                     print(student, choice, cost, CostPenalty)
#                     projects_subset = self.projects.loc[choice]
#                     print(projects_subset)
#                     for adjustment in adjustments:
#                         supervisor = list(adjustment.keys())[0]
#                         change = adjustment[supervisor]
#                         print(supervisor, projects_subset.Section)
#                         assert projects_subset.Section != "Baum"
#                         if projects_subset.Section == supervisor:
#                             print("!")
#                             sys.exit()
# 
# =============================================================================
                

        
        print("\nCostMatrix")
        print(self.CostMatrix)
        

    def Hungarian(self):
        row_ind, col_ind = linear_sum_assignment(self.CostMatrix)
        
        self.Result = pandas.DataFrame(index=self.choices.index, columns=["Project", "Choice #"])
        
        for i in range(len(row_ind)):
            student = self.CostMatrix.index[row_ind[i]]
            project = self.CostMatrix.columns[col_ind[i]]
            project = np.floor(project).astype(np.int64)
            self.Result.at[student, "Project"] = project
            try:
                self.Result.at[student, "Choice #"] = np.where(self.choices.loc[student] == project)[0][0]+1
            except:
                self.Result.at[student, "Choice #"] = -1
            
            
        print(self.Result)
        
    def summary(self):
        print("="*50)
        print("Result summary:")
        print("Number of students that got their first choice:", (self.Result["Choice #"] == 1).sum())
        print("Number of students that got their second choice:", (self.Result["Choice #"] == 2).sum())
        print("Number of students that got their third choice:", (self.Result["Choice #"] == 3).sum())
        print("Number of students that got their fourth choice:", (self.Result["Choice #"] == 4).sum())
        print("Number of students that got their fifth choice:", (self.Result["Choice #"] == 5).sum())
        print("Lowest choice:", self.Result["Choice #"].max())
        
        print("Average choice that each student got:", round(self.Result[self.Result["Choice #"] != -1]["Choice #"].values.astype(np.float64).mean(), 2))
        print("Did any student get assigned a project that wasn't on their list of choices:", (self.Result["Choice #"] == -1).any())

        self.categories = {"green": {"assigned":0, "cap":12}, 
                           "Catalysis (blue)": {"assigned":0, "cap":22}, 
                           "BioNano (orange)": {"assigned":0, "cap":35}, 
                           "MCCB (purple)": {"assigned":0, "cap":28}, 
                           "red": {"assigned":0, "cap":9}}
        
        for project_choice in self.Result["Project"]:
            if project_choice <= 12:
                self.categories["green"]["assigned"]  += 1
            elif project_choice > 12 and project_choice<= 33:
                self.categories["Catalysis (blue)"]["assigned"]  += 1
            elif project_choice > 33 and project_choice<= 57:
                self.categories["BioNano (orange)"]["assigned"]  += 1
            elif project_choice > 57 and project_choice<= 82:
                self.categories["MCCB (purple)"]["assigned"]  += 1
            elif project_choice > 82:
                self.categories["red"]["assigned"]  += 1
        for colour in self.categories:
            print(f"{colour} had capacity for:", self.categories[colour]["cap"],
                  "and got assigned", self.categories[colour]["assigned"],
                  "\t ratio:", self.categories[colour]["assigned"]/self.categories[colour]["cap"], ": 1")
            
        #by-sectiongroup-ratio (MCCB offered 15 : 80, BIONANO X:N etc)
        print("="*50)

        
    def capacity_check(self):
        print("Checking that each supervisor has atleast 1 student")
        print("Checking that no supervisor is oversubscribed")
        supervisors = np.unique(self.projects["Section"])
        print(supervisors)
        for supervisor in supervisors:
            subset = self.projects[self.projects["Section"] == supervisor]
            capacity = subset["Capacity"].sum()
            
            assigned_students = 0
            for p in [x for x in subset.index if x == np.floor(x)]:
                p = int(p)
                chosen = self.Result[self.Result["Project"] == p]
        # =============================================================================
        #         if chosen.shape[0] > 1:
        #             sys.exit()
        # =============================================================================
                #print(chosen)
                assigned_students += chosen.shape[0]
            
            print(supervisor, capacity, assigned_students)
            
            assert capacity >= assigned_students
            
            if assigned_students == 0:
                project_ints = [x for x in subset.index if x == np.floor(x)]
                print(project_ints)
                for p in project_ints:
                    print("p in choices", p in self.DecimalChoices.values)
                    
                if p not in self.DecimalChoices.values: #"Nobody chose this project at all"
                    continue
                ##return {supervisor: 0.1}
            #assert assigned_students > 0
        return None

    
    def save(self):
        self.Result.to_csv("Project-assignment.csv")
        
    def __init__(self):
        self.load_projects()
        self.load_choices()
        self.calc_CostMatrix()
    
if __name__ == "__main__":
    adjustments = []
    spa = StudentProjectAssignment()
    
    spa.Hungarian()
    spa.summary()
    
    sys.exit()
    
    adjustment = spa.capacity_check()
    
    if adjustment is not None:
        adjustments.append(adjustment)
    print(adjustments)
    if len(adjustments) > 0:
        spa.calc_CostMatrix(adjustments)
        
        




# =============================================================================
# # Set up the matplotlib figure
# f, ax = plt.subplots(figsize=(11, 9))
# # Generate a custom diverging colormap
# cmap = sns.diverging_palette(230, 20, as_cmap=True)
# # Draw the heatmap with the mask and correct aspect ratio
# sns.heatmap(CostMatrix.astype(np.float64), cmap=cmap, vmax=HighCost, vmin=0, center=0, square=True, linewidths=.5, cbar_kws={"shrink": .5})
# =============================================================================









