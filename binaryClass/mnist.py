'''
Created on Feb 4, 2015

@author: yuyin
'''
import cPickle, gzip, numpy
import matplotlib.pyplot as plt
import pylab
from numpy.lib.scimath import sqrt
def distImages(a,b):
    return sqrt(((a-b) * (a-b)).sum())
targetDigit = 8  
trainTarget = numpy.zeros(784)
matrTargetNum = 0
matrOtherNum = 0
errorTrain = 0;
errorTest = 0;
trainMatrOther = numpy.zeros(784)
f = gzip.open('mnist.pkl.gz', 'rb')
train_set, valid_set, test_set = cPickle.load(f)
f.close()
for num in range(0,1000):
    if train_set[1][num] == targetDigit:
        trainTarget = trainTarget+train_set[0][num]
        matrTargetNum += 1
    else:
        trainMatrOther = trainMatrOther + train_set[0][num]
        matrOtherNum += 1
trainTarget = (trainTarget/matrTargetNum).reshape(28,28)
trainMatrOther = (trainMatrOther/matrOtherNum).reshape(28,28)
for num in range(0,1000):
    im = train_set[0][num].reshape(28,28)
    if distImages(trainTarget, im) < distImages(trainMatrOther, im):
        if train_set[1][num] != targetDigit:
            errorTrain += 1
    else:
        if train_set[1][num] == targetDigit:
            errorTrain += 1
for num in range(0,1000):
    im = test_set[0][num].reshape(28,28)
    if distImages(trainTarget,im) < distImages(trainMatrOther, im):
        if test_set[1][num] != targetDigit:
            errorTest += 1
    else:
        if test_set[1][num] == targetDigit:
            errorTest += 1
print errorTrain,errorTest
print train_set[1][0]
# plt.subplot(1,2,1)
# plt.imshow(trainTarget, pylab.gray())
# plt.subplot(1,2,2)
# plt.imshow(trainMatrOther, pylab.gray())
# plt.savefig("mini7.png")
plt.show()


