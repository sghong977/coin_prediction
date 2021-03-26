import numpy as np
import pandas as pd
import matplotlib as mpl
import os
#if os.environ.get('DISPLAY','') == '':
#    print('no display found. Using non-interactive Agg backend')
#    mpl.use('Agg')
import matplotlib.pyplot as plt

import datetime

from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split


import random
from pylab import mpl, plt
plt.style.use('seaborn')
mpl.rcParams['font.family'] = 'serif'

#from pandas import datetime
import math, time
import itertools
from sklearn import preprocessing
import datetime
from operator import itemgetter
from sklearn.metrics import mean_squared_error
from math import sqrt
import torch
import torch.nn as nn
from torch.autograd import Variable


def load_raw(visualization=False):
    print("preparing raw data...")

    # read Data
    df = pd.read_csv("./bitstampUSD_1-min_data_2012-01-01_to_2020-12-31.csv")
    df['price'] = (df['High']+ df['Low'])/2
    df.drop(['Open','Close','Volume_(BTC)','Volume_(Currency)', 'Weighted_Price','High','Low'],axis=1, inplace=True)

    df['Timestamp'] = pd.to_datetime(df['Timestamp'],unit='s')
    df = df.set_index('Timestamp')
    #df = df.resample('6H').mean()
    df = df.loc['2020-11-01':]
    df = df.dropna()

    # show
    if visualization ==True:
        plt.figure(figsize=(20,10))
        plt.plot(df)
        plt.title('Bitcoin price',fontsize=20)
        plt.xlabel('year',fontsize=15)
        plt.ylabel('price',fontsize=15)
        plt.savefig("data.png")
    return df

"""
D: seq_len
"""

#normalizing price
scaler = MinMaxScaler()

def load_data(D=50):
    global scaler

    df = load_raw()
    print("build dataset...")
    
    price = scaler.fit_transform(np.array(df['price']).reshape(-1,1))
    df['price'] = price

    # split train and test data
    X_l = []
    y_l = []
    N = len(df)
    for i in range(N-D-1):
        X_l.append(df.iloc[i:i+D].to_numpy())
        y_l.append(df.iloc[i+D].to_numpy())

    #(12965, 50, 1) (12965, 1)
    X = np.array(X_l)
    y = np.array(y_l)

    X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2, random_state= 100)
    X_train = torch.from_numpy(X_train).type(torch.Tensor)
    X_test = torch.from_numpy(X_test).type(torch.Tensor)
    y_train = torch.from_numpy(y_train).type(torch.Tensor)
    y_test = torch.from_numpy(y_test).type(torch.Tensor)
    print("Train data:", len(X_train), X_train.shape)
    print("Test data:", len(X_test), X_test.shape)

    return [X_train, y_train, X_test, y_test]

#X_train, y_train, X_test, y_test = load_data()
#print(len(X_train))

