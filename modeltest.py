#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 26 13:56:53 2017

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
#dat=copy.deepcopy(data)

# reshape into X=t and Y=t+1

trainX, trainY = create_dataset(data, dataY)
#testX, testY = create_dataset(test, look_back)

mx=np.amax(dataY)
mn=np.amin(dataY)

trainY=trainY.reshape(-1,1)


# normalize the dataset
scaler = MinMaxScaler(feature_range=(0, 1))
trainX = scaler.fit_transform(trainX)
trainY = scaler.fit_transform(trainY)

from keras.layers import Input, Embedding, LSTM, Dense
from keras.models import Model

# Headline input: meant to receive sequences of 100 integers, between 1 and 10000.
# Note that we can name any layer by passing it a "name" argument.
main_input = Input(shape=(100,), dtype='int32', name='main_input')

# This embedding layer will encode the input sequence
# into a sequence of dense 512-dimensional vectors.
x = Embedding(output_dim=512, input_dim=10000, input_length=100)(main_input)

# A LSTM will transform the vector sequence into a single vector,
# containing information about the entire sequence
lstm_out = LSTM(32)(x)

auxiliary_output = Dense(1, activation='sigmoid', name='aux_output')(lstm_out)

auxiliary_input = Input(shape=(5,), name='aux_input')
x = keras.layers.concatenate([lstm_out, auxiliary_input])

# We stack a deep densely-connected network on top
x = Dense(64, activation='relu')(x)
x = Dense(64, activation='relu')(x)
x = Dense(64, activation='relu')(x)

# And finally we add the main logistic regression layer
main_output = Dense(1, activation='sigmoid', name='main_output')(x)

model = Model(inputs=[main_input, auxiliary_input], outputs=[main_output, auxiliary_output])

model.compile(optimizer='rmsprop', loss='binary_crossentropy',
              loss_weights=[1., 0.2])

model.fit([headline_data, additional_data], [labels, labels],
          epochs=50, batch_size=32)


