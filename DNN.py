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
learning_rate = 50

print(data_x.size())
print(data_y.size())


# =============================================================================
# example_2D_list = [list(data_x.shape),
#                    list(data_x.shape),
#                    [85, 5]]
# list_to_tensor = torch.tensor(example_2D_list)
# print("Our New 2D Tensor from 2D List is: ", list_to_tensor)
# =============================================================================


class IntegerActivation(nn.Module):
    def __init__(self):
        super(IntegerActivation, self).__init__()

    def forward(self, x):
        rounded_output = torch.absolute(torch.round(x))  # Round to nearest whole integer
        return rounded_output

model = nn.Sequential(nn.Conv1d(85, 10,5),
                      nn.Linear(in_features=6, out_features=10),
                      nn.ReLU(),
                      nn.Flatten(),
# =============================================================================
#                       nn.Linear(in_features=100, out_features=85),
#                       nn.ReLU(),
#                       nn.Linear(in_features=85, out_features=85),
#                       nn.ReLU(),
# =============================================================================
                      nn.Linear(in_features=100, out_features=170),
                      nn.Linear(in_features=170, out_features=85),

                      IntegerActivation()
                      )

# =============================================================================
# for j, p in enumerate(model.parameters()):
#     p.requires_grad_(False)
# =============================================================================
        
model.to(device)
print(model)

#sys.exit()


class CustomLoss(nn.Module):
    def __init__(self):
        super(CustomLoss, self).__init__()

    def forward(self, prediction):
        #Loss for making all the values unique
# =============================================================================
#         unique_count = torch.unique(prediction).size(0)
#         UniquenessLoss = torch.absolute(unique_count - torch.Tensor([88]))*10
#         
#         #Loss for making sure all the selections add up to the right amount
#         SUM = 3916.0
#         SUMLoss = torch.absolute(prediction.sum() - SUM) 
#         
#         #Loss to penalize any value being below 1 or above 88
#         penalty = torch.where((prediction <= 0) | (prediction >= 88), torch.tensor(1.0), torch.tensor(0.0))
#         RangeLoss = torch.sum(penalty)*100
#         
#         #Make sure it has requires_grad
#         requires_grad = prediction.sum() - prediction.sum()
#         
#         loss = UniquenessLoss  + RangeLoss + requires_grad  + SUMLoss
#         return loss
# =============================================================================
        num_duplicates = prediction.flatten().size(0) - torch.unique(prediction).size(0)
        
        # Use the number of duplicates as the loss value
        loss = num_duplicates
        return loss + prediction.sum() - prediction.sum()
        
        
        
#loss_function = nn.MSELoss()
loss_function = CustomLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.09)
SGD_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.2, patience=100, threshold=0, 
                                                           min_lr = 0.000001, verbose=True)


losses = []
pred_history = np.ndarray((0, 85))
for epoch in range(2000):
    pred_y = model(data_x.reshape(1, 85, 10))
    pred_history = np.vstack((pred_history, pred_y.flatten().detach().numpy()))
    loss = loss_function(pred_y)
    losses.append(loss.item())
    
# =============================================================================
#     if loss.item() > 10:
#         break
# =============================================================================
    
    optimizer.zero_grad()
    loss.backward()
    SGD_scheduler.step(loss)
    # SGD_scheduler._last_lr
    
    optimizer.step()
losses = np.array(losses)
print(losses)

print(pred_history[np.argmin(losses)])
print(losses[np.argmin(losses)])
print("Final learning rate:", SGD_scheduler._last_lr)


plt.plot(losses[100:])
#plt.plot(losses)
plt.ylabel('loss')
plt.xlabel('epoch')
plt.title("Learning rate %f"%(learning_rate))
plt.show()
