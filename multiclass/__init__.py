'''
Created on Feb 11, 2015

@author: yuyin
'''
import cPickle, gzip, numpy
import matplotlib.pyplot as plt
import pylab
import numpy as np
from numpy.lib.scimath import sqrt
 
w = numpy.zeros(785)
allWof = []
goBack = True
allError = []
finalw = []
f = gzip.open('mnist.pkl.gz', 'rb')
train_set, valid_set, test_set = cPickle.load(f)
f.close()

def training(num):
    global allWof, errorNum,finalw
    goBack = True
    w = numpy.zeros(785)
    epochw = []
    while(goBack):
        goBack = False
        for i in range(0,100):
            x = np.append(1,train_set[0][i])
            if sum(w*x) > 0:
                if train_set[1][i] != num:
                    w = w - x;
                    epochw.append(w)
                    goBack = True
                    break
            else:
                if train_set[1][i] == num:
                    w  = w + x;
                    epochw.append(w)
                    goBack = True
                    break
    allWof.append(epochw)
    finalw.append(w)
# for target in range(10):
#     allError
#     training(target)
#     errorNum = []
#     for i in allWof[target]:
#         error = 0
#         for j in range(0,1000):
#             x = np.append(1,test_set[0][j])
#             if sum(i*x) > 0: 
#                 if test_set[1][j] != target:
#                     error += 1
#             else:
#                 if test_set[1][j] == target:
#                     error += 1 
#         errorNum.append(error) 
#     allError.append(errorNum)
#     plt.subplot(5,2,target)
#     plt.plot(range(len(allWof[target])),allError[target])
# plt.savefig("100.png")
# plt.show()

# for target in range(10):
#     allError
#     training(target)
#     errorNum = []
#     for i in allWof[target]:
#         error = 0
#         for j in range(0,1000):
#             x = np.append(1,train_set[0][j])
#             if sum(i*x) > 0: 
#                 if train_set[1][j] != target:
#                     error += 1
#             else:
#                 if train_set[1][j] == target:
#                     error += 1 
#         errorNum.append(error) 
#     allError.append(errorNum)
#     plt.subplot(5,2,target)
#     plt.plot(range(len(allWof[target])),allError[target])
# plt.savefig("100.png")
# plt.show()

# def epoch():
#     errorPlot = []
#     global allWof, errorNum,finalw
#     goBack = True
#     epochTime = 0
#     w = [numpy.zeros(785),numpy.zeros(785),numpy.zeros(785),numpy.zeros(785),
#          numpy.zeros(785),numpy.zeros(785),numpy.zeros(785),numpy.zeros(785),
#          numpy.zeros(785),numpy.zeros(785)]
#     while(goBack):
#         goBack = False
#         for i in range(0,1000):
#             x = np.append(1,train_set[0][i])
#             for j in range(10):
#                 if sum(w[j]*x) > 0:
#                     if train_set[1][i] != j:
#                         w[j] = w[j] - x;
#                         goBack = True
#                 else:
#                     if train_set[1][i] == j:
#                         w[j] = w[j] + x;
#                         goBack = True
#             if goBack == True:
#                 break
#         errorPlot.append(errorRate(w))
#         epochTime+=1
#     print epochTime
#     plt.plot(range(epochTime),errorPlot)
#     plt.savefig("q4_1000.png")
#     plt.show()
#     finalw = w
# def inferenceAll(epochw,current,num):
#     x = np.append(1,test_set[0][current])
#     if sum(epochw[num]*x) > 0:
#         return 1
#     else:
#         return -1
# def oaaDecision(epochw,current):
#     inClass = []
#     notInClass = []
#     x = np.append(1,test_set[0][current])
#     for i in range(10):
#         if inferenceAll(epochw,current,i) == 1:
#             inClass.append((i,sum(epochw[i]*x)))
#         else:
#             notInClass.append((i,sum(epochw[i]*x)))
#     if len(inClass) == 0:
#         q = sorted(notInClass, key = lambda x:x[1],reverse = True)
#         return q[0][0]
#     else:
#         q = sorted(inClass, key = lambda x:x[1],reverse = True)
#         return q[0][0]
# def errorRate(epochw):
#     error = 0;
#     for i in range(1000):
#         decision = oaaDecision(epochw, i)
#         lable = test_set[1][i]
#         if decision != lable:
#             error+=1;
#     return float(error)/1000
# epoch()

def inference(current,num):
    global finalw
    x = np.append(1,test_set[0][current])
    if sum(finalw[num]*x) > 0:
        return 1
    else:
        return -1
def finalDecision(current):
    global finalw
    inClass = []
    notInClass = []
    x = np.append(1,test_set[0][current])
    for i in range(10):
        if inference(current, i) == 1:
            inClass.append((i,sum(finalw[i]*x)))
        else:
            notInClass.append((i,sum(finalw[i]*x)))
    if len(inClass) == 0:
        q = sorted(notInClass, key = lambda x:x[1],reverse = True)
        return q[0][0]
    else:
        q = sorted(inClass, key = lambda x:x[1],reverse = True)
        return q[0][0]
for target in range(10):
    training(target)
confusionMatrix = numpy.zeros(100).reshape(10,10)
errorNum = [0,0,0,0,0,0,0,0,0,0,0]
for i in range(1000):
    decision = finalDecision(i)
    lable = test_set[1][i]
    if decision != lable:
        errorNum[10] += 1
        errorNum[decision] += 1
        confusionMatrix[decision,lable]+=1
print confusionMatrix