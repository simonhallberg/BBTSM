import numpy as np
import matplotlib.pyplot as plt
import math
from keras.layers import Input, LSTM
from sklearn.metrics import mean_squared_error
import cPickle as pickle
import copy
import time

from keras.models import Model
t1=time.time()



# fix random seed for reproducibility
np.random.seed(7)

data = pickle.load( open( "savedata200.p", "rb" ) )
dataY = pickle.load( open( "savedataY200.p", "rb" ) )
datatest = pickle.load( open( "savedatatest200.p", "rb" ) )
dataYtest = pickle.load( open( "savedataYtest200.p", "rb" ) )

data=data[:,4:]
datatest=datatest[:,4:]




mintot=min(np.amin(dataY),np.amin(dataYtest))
maxtot=min(np.amax(dataY),np.amax(dataYtest))

normtrain=copy.deepcopy(dataY)
normtrain=(normtrain-mintot)/(maxtot-mintot)
normtest=copy.deepcopy(dataYtest)
normtest=(normtest-mintot)/(maxtot-mintot)
normtestparams=copy.deepcopy(datatest)
normtestparams=(normtestparams-np.amin(normtestparams))/\
(np.amax(normtestparams)-np.amin(normtestparams))
normtrainparams=copy.deepcopy(data)
normtrainparams=(normtrainparams-np.amin(normtrainparams))/\
(np.amax(normtrainparams)-np.amin(normtrainparams))

normtrainparams=np.reshape(normtrainparams, 
                    (normtrainparams.shape[0], normtrainparams.shape[1],1))
normtestparams=np.reshape(normtestparams, 
                    (normtestparams.shape[0], normtestparams.shape[1],1))

t2=time.time()-t1




input1 = Input(shape=(normtrainparams.shape[1],1))

lstm = LSTM(normtrain.shape[1], return_sequences=True)(input1)
lstm = LSTM(normtrain.shape[1], return_sequences=False)(lstm)

model = Model(inputs = input1, outputs = lstm)
model.compile(loss='mean_squared_error', optimizer='adam')

model.fit(normtrainparams, normtrain,
          epochs=300)
t3=time.time()-t2-t1


trainPredict = model.predict(normtrainparams)
testPredict = model.predict(normtestparams)

trainPredict = mintot+trainPredict*(maxtot-mintot)
testPredict = mintot+testPredict*(maxtot-mintot)


# calculate root mean squared error
trainScore = math.sqrt(mean_squared_error(dataY, trainPredict))
print('Train Score: %.2f RMSE' % (trainScore))
testScore = math.sqrt(mean_squared_error(dataYtest, testPredict))
print('Test Score: %.2f RMSE' % (testScore))


#plotting the best fitted time series
err=100000
minvec1=[]
minvec2=[]
for i in range(len(dataYtest[:,0])):
    temp=math.sqrt(mean_squared_error(dataYtest[i,:], testPredict[i,:]))
    if temp<err:
        err=temp
        minvec1=dataYtest[i,:]
        minvec2=testPredict[i,:]
        idx=i
    minvec1=np.array(minvec1)
    minvec2=np.array(minvec2)
   
plt.plot(minvec1)
plt.plot(minvec2)
plt.show()



fig, ax = plt.subplots(figsize=(11,8.5))
ax.plot(minvec1, label='Test Data')
ax.plot(minvec2, label='Test Prediction')
plt.ylabel('Population Count', size=15)
plt.xlabel('Time Step', size=15)


# Now add the legend with some customizations.
legend = ax.legend(loc='upper center', shadow=True)

# The frame is matplotlib.patches.Rectangle instance surrounding the legend.
frame = legend.get_frame()
frame.set_facecolor('0.90')

# Set the fontsize
for label in legend.get_texts():
    label.set_fontsize(15)

for label in legend.get_lines():
    label.set_linewidth(1.5)  # the legend line width
#plt.show()
#plt.gca().set_position([0, 0, 1, 1])
plt.savefig("test.jpg")



