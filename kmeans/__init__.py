import cPickle, gzip
from numpy import *  
import time  
import pylab
import matplotlib.pyplot as plt
from sklearn.preprocessing import normalize
def unpickle(im_file):
    import cPickle
    fo = open(im_file, 'rb')
    im_dict = cPickle.load(fo)
    fo.close()
    return im_dict
def rgb2gray(rgb):
    return dot(rgb[...,:3], [0.299, 0.587, 0.144])

def grayscale_img(dic):
    data = []
    for im in dic["data"]:
        img = zeros((32, 32, 3), dtype=float)
        img[:,:,0] =  im[0:1024].reshape(32,32)
        img[:,:,1] = im[1024:2048].reshape(32,32)
        img[:,:,2] = im[2048:3072].reshape(32,32)
        gray = rgb2gray(img).reshape(1024)
        data.append(gray)
    return data
def divide_patches(train_imgs):
    after_patches = []
    for img in train_imgs:
        patches = [0,0,0]
        patches[0] = img[0:1024].reshape(32,32)
        patches[1] = img[1024:2048].reshape(32,32)
        patches[2] = img[2048:3072].reshape(32,32) 
        normalize(patches[0][:,newaxis], axis=0).ravel()
        normalize(patches[1][:,newaxis], axis=0).ravel()
        normalize(patches[2][:,newaxis], axis=0).ravel()
        after_patches.append(patches[0])
        after_patches.append(patches[1])
        after_patches.append(patches[2])
    return after_patches
img_file = "data_batch_1"
dic = unpickle(img_file)
trainingSize = 100;
train_imgs = dic["data"][:trainingSize]
train_labels = dic["labels"][:trainingSize]

grayi = zeros((32,32,3), dtype = float)
grayi[:,:,0] = train_imgs[0][0:1024].reshape(32,32)
grayi[:,:,1] = train_imgs[0][1024:2048].reshape(32,32)
grayi[:,:,2] = train_imgs[0][2048:3072].reshape(32,32)
gray = rgb2gray(grayi).reshape(32,32)
name = unpickle("batches.meta")
labelname = name["label_names"]
print labelname[train_labels[0]]
print train_labels[0]
iterationTimes = 100;
def euclDistance(vector1, vector2):  
    return sqrt(sum(power(vector2 - vector1, 2)))  
# init centroids with random samples  
def initCentroids(dataSet, k):  
    numSamples, dim = dataSet.shape  
    centroids = zeros((k, dim))  
    for i in range(k):  
        index = int(random.uniform(0, numSamples))  
        centroids[i, :] = dataSet[index, :]  
    return centroids
# k-means cluster  
def kmeans(dataSet, k):  
    countIt = 0;
    numSamples = dataSet.shape[0]  
    clusterAssment = mat(zeros((numSamples, 2)))  
    clusterChanged = True  
    ## step 1: init centroids  
    centroids = initCentroids(dataSet, k)
    while clusterChanged:  
        countIt+=1
        clusterChanged = False  
        ## for each sample  
        for i in xrange(numSamples):  
            minDist  = 100000.0  
            minIndex = 0  
            for j in range(k):  
                distance = euclDistance(centroids[j, :], dataSet[i, :])  
                if distance < minDist:  
                    minDist  = distance  
                    minIndex = j
            ## step 3: update its cluster  
            if clusterAssment[i, 0] != minIndex:  
                clusterChanged = True  
                clusterAssment[i, :] = minIndex, minDist  
  
        ## step 4: update centroids  
        for j in range(k):  
            pointsInCluster = dataSet[nonzero(clusterAssment[:, 0].A == j)[0]]  
            centroids[j, :] = mean(pointsInCluster, axis = 0)
        print countIt
        if countIt == iterationTimes:
            break
    print 'Congratulations, cluster complete!'  
    return centroids, clusterAssment  
# dataSet = mat(train_set[0][1:1000])
k=10
centroids, clusterAssment = kmeans(train_imgs, k)
print centroids.shape
for i in range(10):
    centroids[i].reshape(32,32,3)
print centroids[1].shape
plt.imshow(centroids) 
plt.show()
accuracy=[[] for i in range(10)]
for i in range(len(clusterAssment)):
    preLabel = int(clusterAssment[i,0])
    accuracy[preLabel].append(train_labels[i])
print accuracy
from collections import Counter
acc = 0;
for i in range(10):
    most_common,num_most_common = Counter(accuracy[i]).most_common(1)[0]
    print most_common,num_most_common
    acc += num_most_common
print float(acc)/trainingSize
# plt.subplot(5,2,1)
# plt.imshow(centroids[1,:].reshape(28,28), pylab.gray())
# plt.subplot(5,2,2)
# plt.imshow(centroids[2,:].reshape(28,28), pylab.gray())
# plt.subplot(5,2,3)
# plt.imshow(centroids[3,:].reshape(28,28), pylab.gray())
# plt.subplot(5,2,4)
# plt.imshow(centroids[4,:].reshape(28,28), pylab.gray())
# plt.subplot(5,2,5)
# plt.imshow(centroids[5,:].reshape(28,28), pylab.gray())
# plt.subplot(5,2,6)
# plt.imshow(centroids[6,:].reshape(28,28), pylab.gray())
# plt.subplot(5,2,7)
# plt.imshow(centroids[7,:].reshape(28,28), pylab.gray())
# plt.subplot(5,2,8)
# plt.imshow(centroids[8,:].reshape(28,28), pylab.gray())
# plt.subplot(5,2,9)
# plt.imshow(centroids[9,:].reshape(28,28), pylab.gray())
# plt.subplot(5,2,10)
# plt.imshow(centroids[0,:].reshape(28,28), pylab.gray())
# plt.show()
print centroids