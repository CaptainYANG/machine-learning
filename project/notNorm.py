import numpy as np
from matplotlib.pyplot import gray
import matplotlib.pyplot as plt
import matplotlib
import random


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
def divide_patches(imgs, w, s):
    patches = []
    for img in imgs:
        stride = s
        cur_x =  0
        cur_y =  0
        imgc = img.reshape(3,32,32)
        while (cur_y <= 32-w):
            while(cur_x <= 32-w):
                patch = imgc[:,cur_x:(cur_x+w),cur_y:(cur_y+w)]
                cur_x += stride
                '''ToTO'''
                
                patch =  patch.reshape(w * w * 3)
                patches.append(patch)
            cur_x = 0
            cur_y += stride
    return patches
def divide_image(img, w, s):
    patches = []
    stride = s
    cur_x =  0
    cur_y =  0
    imgc = img.reshape(3,32,32)

    while (cur_y <= 32-w):
        while(cur_x <= 32-w):
            patch = imgc[:,cur_x:(cur_x+w),cur_y:(cur_y+w)]
            cur_x += stride
            patch =  patch.reshape(w * w * 3)
            patches.append(patch)
        cur_x = 0
        cur_y += stride
    return patches

    
if __name__ == "__main__":
    import datetime
    starttime = datetime.datetime.now()
    
    
    np.set_printoptions(threshold=np.nan) 
    img_file = "data_batch_1"
    dic = unpickle(img_file)
    train_num = 2000
    total_num = train_num+500
    total_imgs = dic["data"][:total_num]
    total_labels = dic["labels"][:total_num]
    
    clusters =  100
    w = 8
    s = 16
    samples = 2000
    
    
    
    ''' to divide into patch '''
    X =  divide_patches(total_imgs,w,s)
    X = random.sample(X,samples)
    print len(X)
    
    ''' kMeans '''
    kmeans = KMeans(n_clusters = clusters)
    kmeans.fit(X)
     
    centroids = kmeans.cluster_centers_
    k_labels = kmeans.labels_ 
    print ("kmeans over")
     
     
    ''' patches - f'''
    labels2 = []
    new_imgs = []
    for img in total_imgs:
        img_patches = divide_image(img,w,s)
        new_img = np.zeros(0)
        for patch in img_patches:
            label = kmeans.predict(patch)[0]
            labels2.append(label)
            new_patch = np.zeros(clusters)
            new_patch[label] = 1
            new_img = np.concatenate((new_img,new_patch),axis = 0)
        new_imgs.append(new_img)
        
 
 
    ''' perceptron for predicting'''
    train = new_imgs[:train_num]
    train_label = total_labels[:train_num]
    clf = Perceptron(n_iter=100).fit(train, train_label)
    test_img  = new_imgs[train_num:total_num]
    test_labels = total_labels[train_num:total_num]
       
   
    right = 0
    for i in range(len(test_img)):
        Z = clf.predict(test_img[i])
        if Z == test_labels[i]:
            right = right +1 
    print right/float(len(test_img))
    
    endtime = datetime.datetime.now()
    print (endtime - starttime).seconds
    
    
#     test_img  = train
#     test_labels = train_label
#        
#     right = 0
#     for i in range(len(test_img)):
#         Z = clf.predict(test_img[i])
#         if Z == test_labels[i]:
#             right = right +1 
#     print right/float(len(test_img))
#     
#     endtime = datetime.datetime.now()
#     print (endtime - starttime).seconds
