# -*- coding: utf-8 -*-
"""
Created on Wed May  3 15:24:47 2023

@author: rkb19187
"""

import numpy as np
import pandas, sys
import matplotlib.pyplot as plt
from sklearn.metrics import euclidean_distances
import torch
import torch.nn as nn

projects = choices = pandas.read_csv("Project List by Section.csv", index_col=0)
choices = pandas.read_csv("Project-Data.csv", index_col=0)
choices.columns = [int(x) for x in choices.columns]
Result = pandas.DataFrame(index=choices.index, columns=["Project"])
#print(choices)
nStudents = choices.shape[0]

MaxScore = choices.shape[0] #[1/x for x in [1,1,1,1,....]]


indices = np.arange(0, projects.shape[0])
Result["Project"] = projects.index[np.random.choice(indices, choices.shape[0], replace=False)]

choices = choices.values.astype(np.float32)

#device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu")
print(device)



data_x = torch.from_numpy(choices)
#target is fake, dont need it for this custom loss function
data_y = torch.from_numpy(Result["Project"].values.reshape(-1,1).astype(np.float32))

n_input = data_x.shape[1]
n_hidden = 15
n_out    = data_y.shape[1]
batch_size = 2
learning_rate = 0.01

print(data_x.size())
print(data_y.size())

model = nn.Sequential(nn.Linear(n_input, n_hidden),
                      nn.ReLU(),
                      nn.Linear(n_hidden, n_out),
                      nn.ReLU())

# =============================================================================
# for j, p in enumerate(model.parameters()):
#     p.requires_grad_(False)
# =============================================================================
        
model.to(device)
print(model)



class CustomLoss(nn.Module):
    def __init__(self, choices):
        super(CustomLoss, self).__init__()
        self.choices = choices
        
    def forward(self, prediction, target):
        #target is fake, dont need it for this custom loss function
        #prediction = torch.round(prediction)
        #print("prediction:", prediction[:5])
        #print("choices:", self.choices[:5])
        
        print(prediction)
        
        self.prediction = prediction
        
        v = euclidean_distances(loss_function.prediction.detach().numpy().flatten().reshape(-1,1), loss_function.prediction.detach().numpy().flatten().reshape(-1,1))
        v[np.diag_indices(v.shape[0])] = 100
        
        
        
        #use the target Tensor as the score board, because it has requires grad
        score = 100
        for i in range(prediction.shape[0]):
            for j in range(target.shape[1]):
                if i == 0:
                    print(i,j, prediction[i], self.choices[i][j])
                if prediction[i] == self.choices[i][j]:
                    target[i] += 100
                    #score += 10
                else:
                    target[i] -1
        
        if torch.mean(target) == 0:
            target[0] = 1
        #print(torch.mean(prediction))
        return 100/target.mean()
        #return 100/prediction.mean()

#loss_function = nn.MSELoss()
loss_function = CustomLoss(choices)
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)


losses = []
for epoch in range(10):
    pred_y = model(data_x)
    loss = loss_function(pred_y, data_y)
    losses.append(loss.item())

    model.zero_grad()
    loss.backward()

    optimizer.step()
    
#print(losses)

plt.plot(losses[1:])
plt.ylabel('loss')
plt.xlabel('epoch')
plt.title("Learning rate %f"%(learning_rate))
plt.show()
