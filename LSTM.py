#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 16 15:34:07 2017

@author: simon
"""
import numpy as np
import matplotlib.pyplot as plt
import pandas
import math
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
import pickle
import copy


# convert an array of values into a dataset matrix
def create_dataset(dat1, dat2):
    datX=dat1.tolist()
    #datY=dat2.tolist()
    dataX, dataY = [], []
    for j in range(len(dat1)):
        for i in range(len(dat2[0,:])-2):
            #b=copy.copy(datX[j])
            dataX.append(copy.copy(datX[j]))
            #a = copy.copy(dat2[j,i])
            dataX[-1].append(copy.copy(dat2[j,i]))
            dataY.append(dat2[j,i + 1])
    return np.array(dataX), np.array(dataY)



# fix random seed for reproducibility
#np.random.seed(7)

data = pickle.load( open( "savedata.p", "rb" ) )
dataY = pickle.load( open( "savedataY.p", "rb" ) )
datatest = pickle.load( open( "savedatatest.p", "rb" ) )
dataYtest = pickle.load( open( "savedataYtest.p", "rb" ) )
#dat=copy.deepcopy(data)







#for i in range(0,len(data)):
#datafirst=data[0][1]
#data=data.reshape(-1, 1)




# reshape into X=t and Y=t+1

trainX, trainY = create_dataset(data, dataY)
testX, testY = create_dataset(datatest, dataYtest)

trainY=trainY.reshape(-1,1)
testY=testY.reshape(-1,1)

#reshape data so that the vector gets the right dimensions
"""
for i in range(len(data)):
    trainY[i][1]=trainY[i][1].reshape(-1,1)
    """

# normalize the dataset
scaler = MinMaxScaler(feature_range=(0, 1))
trainX = scaler.fit_transform(trainX)
trainY = scaler.fit_transform(trainY)
testX = scaler.fit_transform(testX)
testY = scaler.fit_transform(testY)

# reshape input to be [samples, time steps, features]
#trainX = np.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))

# create and fit the LSTM network
"""
model = Sequential()
model.add(LSTM(4, input_shape=(1, look_back)))
model.add(Dense(1))
model.compile(loss='mean_squared_error', optimizer='adam')
model.fit(trainX, trainY, epochs=100, batch_size=1, verbose=2)
"""



#trainX = np.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))

#create numpy array matrix: testmat=numpy.array([[1,2,3],[4,5,6]])

#get element value in the numpy array: data[0][1][0][0]
# reshape input to be [samples, time steps, features]
trainX = np.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))
testX = np.reshape(testX, (testX.shape[0], 1, testX.shape[1]))
# create and fit the LSTM network
model = Sequential()
model.add(LSTM(4, input_shape=(1,8)))
model.add(Dense(1))
model.compile(loss='mean_squared_error', optimizer='adam')
model.fit(trainX, trainY, epochs=100, batch_size=1, verbose=2)
# make predictions
trainPredict = model.predict(trainX)
testPredict = model.predict(testX)
# invert predictions
trainPredict = scaler.inverse_transform(copy.deepcopy(trainPredict))
#nsamples, nx, ny = trainY.shape
#trainY2 = trainY.reshape((nsamples,nx*ny))
trainY = scaler.inverse_transform(copy.deepcopy(trainY))
testPredict = scaler.inverse_transform(copy.deepcopy(testPredict))
testY = scaler.inverse_transform(copy.deepcopy(testY))
# calculate root mean squared error
trainScore = math.sqrt(mean_squared_error(trainY[:,0], trainPredict[:,0]))
print('Train Score: %.2f RMSE' % (trainScore))
testScore = math.sqrt(mean_squared_error(testY[:,0], testPredict[:,0]))
print('Test Score: %.2f RMSE' % (testScore))
"""
# shift train predictions for plotting
trainPredictPlot = np.empty_like(dataset)
trainPredictPlot[:, :] = np.nan
trainPredictPlot[1:len(trainPredict)+1, :] = trainPredict
# shift test predictions for plotting
testPredictPlot = np.empty_like(dataset)
testPredictPlot[:, :] = np.nan
testPredictPlot[len(trainPredict)+2+1:len(dataset)-1, :] = testPredict
# plot baseline and predictions
plt.plot(scaler.inverse_transform(dataset))
plt.plot(trainPredictPlot)
plt.plot(testPredictPlot)
plt.show()
"""

#TODO:
#1. find out how to input exogenous feautures to a LSTM
#2. try to do the multivariate example with the data I have
#3. try a regular supervised learning problem without caring about the sequence
#4. find out if parameters can be added in the input in the model aldready as it is.
#5. look up the merge feature in LSTM
#6. look at my saved bookmarks, some good findings
#7. auxilary input seems to be the name for it

#np.amin(nparrays) finds min value, max for max
#blablabla=np.concatenate((data,dataY), axis=1) concatenates the paramteters with the time series





    

