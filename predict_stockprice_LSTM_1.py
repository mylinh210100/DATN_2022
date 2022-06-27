from ast import increment_lineno
from datetime import datetime
import imp
from tkinter import font
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.axes_style('whitegrid')
plt.style.use('fivethirtyeight')

import pandas_datareader as data
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout


df = data.DataReader('TSLA', data_source='yahoo', start ='2014-01-01', end='2021-12-30')
print(df)

"""
    #mo hinh hoa data 'close' day
plt.figure(figsize=(12,6))
plt.title('Close Price History of TSLA')
plt.plot(df['Close'])
plt.xlabel('Time', fontsize=18)
plt.ylabel('Close Price (USD)', fontsize=18)
plt.show()

    #mo hinh hoa data "volume"
plt.figure(figsize=(12,6))
plt.title('Sales for Volume of TSLA')
plt.plot(df['Volume'])
plt.xlabel('Time', fontsize=18)
plt.ylabel('Volume', fontsize=18)
plt.show()


    # We'll use pct_change to find the percent change for each day
df['Daily return'] = df['Close'].pct_change()
#print(df['Daily return'])

    #mo hinh hoa cot Daily return
plt.figure(figsize=(12,7))
plt.title('Daily return of TSLA')
df['Daily return'].hist(bins=50)
plt.xlabel('Time', fontsize=18)
plt.ylabel('Daily return')
plt.show()
"""

    # Tao 1 DF voi chi minh cot Close
dataf = df.filter(['Close'])
    # Convert DF to a numpy array
dataset = dataf.values

    # Get the number to train the model on 85%
training_data_len = int(np.ceil( len(dataset) * 0.85 ))
#print(training_data_len) #1731

    # Scale the data 
    # use Scikit- Learn’s MinMaxScaler and scale our dataset to numbers between zero and one.
scaler = MinMaxScaler(feature_range=(0,1))
scaled_data = scaler.fit_transform(dataset)
print(scaled_data)

    # tao dataset de training  
    # Create the scaled training data set
train_data = scaled_data[0:training_data_len, :] #get data from 0 to 1731, and all column
    # Split the data into x_train and y_train datasets
x_train = []
y_train = []

    #Creating a data structure with 60 time-steps and 1 output
for i in range(60, len(train_data)):
    x_train.append(train_data[i-60:i, 0])
    y_train.append(train_data[i, 0])
    if i<= 61:
        print(x_train)
        print(y_train)
        print()

    # Convert the x_train and y_train to numpy arrays 
x_train, y_train = np.array(x_train), np.array(y_train)

    # Reshape the data
x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
#print(x_train.shape) (1653,60,1)


    # Build the LSTM model
model = Sequential()
    # 1st layer
model.add(LSTM(128, return_sequences=True, input_shape=(x_train.shape[1], 1))) # 50 neuron
    # 2nd layer
model.add(LSTM(64, return_sequences=False))

    # output layer
model.add(Dense(25)) #25 noron
model.add(Dense(1))

    # Compile the RNN
model.compile(optimizer='adam', loss='mean_squared_error')

    # Fitting the RNN to Train the model
model.fit(x_train, y_train, batch_size=1, epochs=2)

    # Create the testing data set
    #Create a new array containing scaled values from  to
test_data = scaled_data[training_data_len - 60: , :]

    # Create the data sets x_test and y_test
x_test = []
y_test = dataset[training_data_len:, :]
for i in range(60, len(test_data)):
    x_test.append(test_data[i-60:i, 0])
    
    # Convert the data to a numpy array
x_test = np.array(x_test)

    # Reshape the data
x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1 ))
print(x_test.shape) #(302,60,1)

    # Get the models predicted price values 
predictions = model.predict(x_test)
predictions = scaler.inverse_transform(predictions)

    # Get the root mean squared error (RMSE) loi trung binh binh phuong goc
rmse = np.sqrt(np.mean(((predictions - y_test) ** 2)))
print(rmse)

    # Plot the data - ve du lieu
train = dataf[:training_data_len]
valid = dataf[training_data_len:]
valid['Predictions'] = predictions

    # Visualize the data
plt.figure(figsize=(16,8))
plt.title('Tesla Stock price prediction')
plt.xlabel('Date', fontsize=18)
plt.ylabel('Close Price (USD)', fontsize=18)
plt.plot(train['Close'])
plt.plot(valid[['Close', 'Predictions']])
plt.legend(['Train', 'Real stock price', 'Predictions stock price'], loc='lower right')
plt.show()

    # Show the valid and predicted prices
print(valid)

