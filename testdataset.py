#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 22 18:26:21 2017

@author: simon
"""
import numpy as np
import copy


blabla1=np.array([[1,2,3], [4,5,6]])
blabla2=np.array([[8,9,10,11,12,13,14], [8,10,12,14,16,18,20]])

blablaX=blabla1.tolist()
dataX, dataY = [], []

for j in range(len(blabla1)):
    for i in range(len(blabla2[0,:])-2):
        b=copy.copy(blablaX[j])
        dataX.append(b)
        a = copy.copy(blabla2[j,i])
        dataX[-1].append(a)
        dataY.append(blabla2[j,i + 1])

#j=0
#i=0   
""" 
b=copy.copy(blablaX[0])   
dataX.append(b)
a = blabla2[0,0]
dataX[-1].append(a)
dataY.append(blabla2[0,1])
"""
