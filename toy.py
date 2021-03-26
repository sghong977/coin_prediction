import numpy as np
import pandas as pd
import matplotlib as mpl
import os
if os.environ.get('DISPLAY','') == '':
    print('no display found. Using non-interactive Agg backend')
    mpl.use('Agg')
import matplotlib.pyplot as plt

import datetime

from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

import keras
from tensorflow.keras import models, layers
from tensorflow.python.client import device_lib
device_lib.list_local_devices()


# read Data
df = pd.read_csv("./bitstampUSD_1-min_data_2012-01-01_to_2020-12-31.csv")
df['price'] = (df['High']+ df['Low'])/2
df.drop(['Open','Close','Volume_(BTC)','Volume_(Currency)', 'Weighted_Price','High','Low'],axis=1, inplace=True)

df['Timestamp'] = pd.to_datetime(df['Timestamp'],unit='s')
df = df.set_index('Timestamp')
df = df.resample('6H').mean()
df = df.dropna()

df.head()

# show
plt.figure(figsize=(20,10))
plt.plot(df)
plt.title('Bitcoin price',fontsize=20)
plt.xlabel('year',fontsize=15)
plt.ylabel('price',fontsize=15)
plt.savefig("data.png")


#normalizing price
scaler = MinMaxScaler()
price = scaler.fit_transform(np.array(df['price']).reshape(-1,1))
df['price'] = price

# split train and test data
X_l = []
y_l = []
N = len(df)
D = 50
for i in range(N-D-1):
    X_l.append(df.iloc[i:i+D])
    y_l.append(df.iloc[i+D])
    
X = np.array(X_l)
y = np.array(y_l)

print(X.shape, y.shape)

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2, random_state= 100)

# define model
m_x = layers.Input(shape = X_train.shape[1:])
m_h = layers.LSTM(10)(m_x)
m_y = layers.Dense(1)(m_h)
m = models.Model(m_x,m_y)
#m = keras.utils.multi_gpu_model(m, gpus=2)
m.compile('adam','mse')
m.summary()

# fit
#from tensorflow.keras import backend as K
#K.tensorflow_backend._get_available_gpus()
history = m.fit(X_train, y_train, epochs=500, validation_data=(X_test, y_test),verbose=0)
# Then explore the hist object
history.history #gives you a dictionary
history.epoch #gives you a list

# display
plt.figure(figsize=(15,5))
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.legend(['Train','Test'])
plt.title("The model's evaluation", fontsize=14)
plt.xlabel('Epoch')
plt.xlim(2,500)
plt.ylabel('Loss')
plt.savefig("eval.png")



# price prediction
pred = []

pr = m.predict(np.array(df[-50:]))

pred.append(pr[0])

for i in range(1,50):
    pr = m.predict(np.concatenate((np.array(df[-50+i:]), pred[:]), axis=0))
    pred.append(pr[0])

for i in range(0,250):
    pr = m.predict(np.concatenate(pred[i:],axis=0).reshape(-1,1))
    pred.append(pr[0])

pred = pd.DataFrame(pred)

pred = pred.reset_index()

pred.columns = ['z','price']

pred.drop(['z'],axis=1,inplace=True)

data = pd.concat([df.reset_index().drop('Timestamp',axis=1),pred],ignore_index=True)

plt.figure(figsize=(17,7))
plt.plot(data[-1300:-300])
plt.title("Bitcoin predict",fontsize=20)
plt.text(13200,1,"predict data",fontsize=14)
plt.text(13015,1,"- 2020/12/31",fontsize=14)
plt.plot(data[-300:])
plt.savefig("prediction_output.png")