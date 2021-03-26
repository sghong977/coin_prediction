import numpy as np
import pandas as pd
import matplotlib as mpl
import os
import matplotlib.pyplot as plt
import datetime

from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

import random
from pylab import mpl, plt
plt.style.use('seaborn')
mpl.rcParams['font.family'] = 'serif'

import math, time
import itertools
from sklearn import preprocessing
from operator import itemgetter
from sklearn.metrics import mean_squared_error
from math import sqrt
import torch
import torch.nn as nn
import torch.utils.data
from torch.autograd import Variable

#https://www.kaggle.com/taronzakaryan/predicting-stock-price-using-lstm-model-pytorch

###########################################
from generate_data import load_data, scaler
from plot import plot_result

# Here we define our model as a class
class LSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim):
        super(LSTM, self).__init__()
        self.hidden_dim = hidden_dim
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)

        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        # Initialize hidden state with zeros
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).requires_grad_().to(self.device)
        # Initialize cell state
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).requires_grad_().to(self.device)

        # We need to detach as we are doing truncated backpropagation through time (BPTT)
        # If we don't, we'll backprop all the way to the start even after going through another batch
        out, (hn, cn) = self.lstm(x, (h0.detach(), c0.detach()))

        # Index hidden state of last time step
        # out.size() --> 100, 32, 100
        # out[:, -1, :] --> 100, 100 --> just want last time step hidden states! 
        out = self.fc(out[:, -1, :]) 
        # out.size() --> 100, 10
        return out



############ main ############
seq_len = 300
batch_size = 1024
num_epochs = 20 #n_iters / (len(train_X) / batch_size)

# setup data
X_train, y_train, X_test, y_test = load_data(seq_len)
X_train, y_train, X_test, y_test = X_train, y_train, X_test, y_test


train = torch.utils.data.TensorDataset(X_train,y_train)
test = torch.utils.data.TensorDataset(X_test,y_test)

train_loader = torch.utils.data.DataLoader(dataset=train, 
                                           batch_size=batch_size, 
                                           shuffle=False,
                                           drop_last = True)

test_loader = torch.utils.data.DataLoader(dataset=test, 
                                          batch_size=batch_size, 
                                          shuffle=False,
                                          drop_last = True)

# model
#####################
input_dim = 1
hidden_dim = 32
num_layers = 2 
output_dim = 1


model = LSTM(input_dim=input_dim, hidden_dim=hidden_dim, output_dim=output_dim, num_layers=num_layers)
model = nn.DataParallel(model)
model = model.cuda()
print(model)

loss_fn = torch.nn.MSELoss()
optimiser = torch.optim.Adam(model.parameters(), lr=0.01)


#train
# Train model
#####################

#hist = np.zeros(num_epochs)
model.train()

for t in range(num_epochs):
    for batch_idx, (inputs, targets) in enumerate(train_loader):
        inputs, targets = inputs.cuda(), targets.cuda()
        y_train_pred = model(inputs)

        loss = loss_fn(y_train_pred, targets)
        #hist[t] = loss.item()

        optimiser.zero_grad()
        loss.backward()
        optimiser.step()

    print("Epoch ", t, "MSE: ", loss.item())

#plt.plot(hist, label="Training loss")
#plt.legend()
#plt.savefig("train_loss.png")

##### switch to evaluate mode
model.eval()
y_test_pred = None
y_tests = None
test_inputs = list()
for batch_idx, (inputs, targets) in enumerate(test_loader):
    inputs, targets = inputs.cuda(), targets.cuda()

    # compute output
    y_test_pred_ = model(inputs)
    loss = loss_fn(y_train_pred, targets)

    out = y_test_pred_.cpu().detach().numpy()
    ttt = targets.cpu().detach().numpy()
    if len(test_inputs) == 0:
        y_test_pred = out
        y_tests = ttt
    else:
        print(y_test_pred, y_tests)
        y_test_pred = np.stack(y_test_pred, out)
        y_tests = np.stack(y_tests, ttt)
    test_inputs.append(inputs)

############ EVAL ################



#y_test_pred = model(X_test)

# invert predictions
#y_train_pred = scaler.inverse_transform(y_train_pred)
#y_train = scaler.inverse_transform(y_train)
#y_test_pred = scaler.inverse_transform(y_test_pred)
#y_test = scaler.inverse_transform(y_test)

# calculate root mean squared error
#trainScore = math.sqrt(mean_squared_error(y_train[:,0], y_train_pred[:,0]))
#print('Train Score: %.2f RMSE' % (trainScore))
#testScore = math.sqrt(mean_squared_error(y_test[:,0], y_test_pred[:,0]))
#print('Test Score: %.2f RMSE' % (testScore))


case_num = list(range(0,10))
for c in case_num:
    print("Case:", c)
    print("GT and pred:", y_test[c], y_test_pred[c])

    original = scaler.inverse_transform(test_inputs[c])
    original = [y for x in original for y in x]
    #print("Window:", original)

    plot_result(original, y_test[c], y_test_pred[c], c)


