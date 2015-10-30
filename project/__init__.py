import numpy as np
from matplotlib.pyplot import gray
import matplotlib.pyplot as plt
import matplotlib
from numpy.random import randint


from sklearn.cluster import KMeans
from array import array
from scipy.cluster.hierarchy import centroid
from sklearn import preprocessing
from sklearn.linear_model import Perceptron 
    
def unpickle(im_file):
    import cPickle
    fo = open(im_file, 'rb')
    im_dict = cPickle.load(fo)
    fo.close()
    return im_dict

#####grayscale##########################
def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.299, 0.587, 0.144])

def grayscale_img(dic):
    data = []
    for im in dic["data"]:
        img = np.zeros((32, 32, 3), dtype=float)
        img[:,:,0] =  im[0:1024].reshape(32,32)
        img[:,:,1] = im[1024:2048].reshape(32,32)
        img[:,:,2] = im[2048:3072].reshape(32,32)
        gray = rgb2gray(img).reshape(1024)
        data.append(gray)
    return data
########################################


############Preprocessing###############
def divide_patches(train_imgs):
    after_patches = []
    for img in train_imgs:
        patches = [[0,0,0],[0,0,0],[0,0,0],[0,0,0]]
        img_sqr = [0,0,0] 
        img_sqr[0] =  img[0:1024].reshape(32,32)
        img_sqr[1] =  img[1024:2048].reshape(32,32)
        img_sqr[2] =  img[2048:3072].reshape(32,32)
        
        w = 16
        
        # channels patches
        for c in range(3):
            patches[0][c] =  img_sqr[c][0:w,0:w]
            patches[1][c] =  img_sqr[c][w:2*w,0:w]
            patches[2][c] =  img_sqr[c][0:w,w:2*w]
            patches[3][c] =  img_sqr[c][w:2*w,w:2*w]
            
        for i in range(len(patches)):
            a = patches[i][0].reshape(w*w)
            b = (patches[i][1]).reshape(w*w)
            c =  (patches[i][2]).reshape(w*w)
            pre_patch = np.concatenate((a, b, c), axis=0)
            after_patches.append(pre_patch)
    return after_patches


      
if __name__ == "__main__":
    np.set_printoptions(threshold=np.nan) 
    img_file = "data_batch_1"
    dic = unpickle(img_file)

    total_num = 2000
    total_imgs = dic["data"][:total_num]
    total_labels = dic["labels"][:total_num]
    
#     imgs = np.array([])
    
    #TODO to divide into patch 
    X =  divide_patches(total_imgs)
    #TODO kMeans change the clusters
    clusters =  100
    kmeans = KMeans(n_clusters= clusters)
    kmeans.fit(X)
    
    print ("kmeans over")
    centroids = kmeans.cluster_centers_
    k_labels = kmeans.labels_ 
    # patches- f
    patches = [0,0,0,0]
    imgs = []
    
    i = 0
    while i != len(k_labels):
        for p in range(len(patches)):
            patches[p] = np.zeros(clusters)
            #triangle code
            patches[p][k_labels[i]] = 1
            i = i+1
            #reduce the dimensions 
            new_patch = np.zeros(clusters/4)
            for a in range(len(new_patch)):
                for b in range(4):
                    new_patch[a] =  new_patch[a]+ patches[p][a*4+b]
            patches[p] =  new_patch
        #patches - imgs
        img = np.concatenate((patches[0],patches[1],patches[2],patches[3]),axis = 0)
        imgs.append(img)

    #TODO: perceptron for predicting
    train_num =  1000
    train = imgs[:train_num]
    train_label = total_labels[:train_num]
    clf = Perceptron(n_iter=50).fit(train, train_label)

    test_img  = imgs[train_num:total_num]
    test_labels = total_labels[train_num:total_num]

    right = 0
    for i in range(len(test_img)):
        Z = clf.predict(test_img[i])
        if Z == test_labels[i]:
            right = right +1 
    print right/float(len(test_img))