'''
Created on Feb 11, 2015

@author: yuyin
'''
import cPickle, gzip, numpy
import matplotlib.pyplot as plt
import pylab
import numpy as np
from numpy.lib.scimath import sqrt
targetDigit = 0   
w = numpy.zeros(785)
error = 0;
goBack = True
total = 0;
f = gzip.open('mnist.pkl.gz', 'rb')
train_set, valid_set, test_set = cPickle.load(f)
f.close()
def epoch():
    global goBack, w,total
    goBack =False
    for num in range(0,100):
        x = np.append(1,train_set[0][num])
        total +=1
        if sum(w*x) > 0:
            if train_set[1][num] != targetDigit:
                w = w - x;
                goBack = True
                return w
        else:
            if train_set[1][num] == targetDigit:
                w  = w + x;
                goBack = True
                return w
epochtime = 0;
allW = []
errorNum = []
while(goBack):
    allW.append(epoch())
    epochtime+=1
print epochtime    
print total
for num in range(0,1000):
    x = np.append(1,test_set[0][num])
    if sum(w*x) >= 0: 
        if test_set[1][num] != targetDigit:
            error += 1
    else:
        if test_set[1][num] == targetDigit:
            error += 1
print error
error = 0;
for num in range(1000,1300):
    x = np.append(1,train_set[0][num])
    if sum(w*x) > 0:
        if train_set[1][num] != targetDigit:
            error += 1
    else:
        if train_set[1][num] == targetDigit:
            error += 1
print error
#plt.imshow(im1.reshape(28,28) , pylab.gray())
