import numpy as np
import random
from matplotlib.pyplot import gray
import matplotlib.pyplot as plt
import matplotlib
import datetime
from sklearn import svm
from sklearn.cluster import KMeans
from array import array
from scipy.cluster.hierarchy import centroid
from sklearn import preprocessing
from sklearn.linear_model import Perceptron 
starttime = datetime.datetime.now()   
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
def normalize(vector):
    if np.std(vector) == 0 :
        new_vector = np.zeros(vector.shape)
    else :
        new_vector = (vector - np.mean(vector))/np.std(vector)
    return new_vector
def euclDistance(vector1, vector2):  
    return sum(np.power(vector2 - vector1, 2))
############Preprocessing###############
def divide_patches(imgs, w, s):
    patches = []
    normPatches = []
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
                normPatch = normalize(patch)
                normPatch = normPatch.reshape(w*w*3)
                normPatches.append(normPatch)
            cur_x = 0
            cur_y += stride
    return normPatches
def divide_image(img, w, s):
    patches = []
    stride = s
    cur_x =  0
    cur_y =  0
    imgc = img.reshape(3,32,32)
    normPatches = []
    while (cur_y <= 32-w):
        while(cur_x <= 32-w):
            patch = imgc[:,cur_x:(cur_x+w),cur_y:(cur_y+w)]
            cur_x += stride
            normPatch = normalize(patch)
            normPatch = normPatch.reshape(w*w*3)
            normPatches.append(normPatch)
        cur_x = 0
        cur_y += stride
    return normPatches

def my_svm (train_x, train_y, test_x, test_y) :
    '''SVM classifier instead of previous perception classifier'''
    clf = svm.SVC(kernel='rbf',gamma=0.7,C = 1.0)
    clf.fit(train_x, train_y)
    right = 0.0
    for i in range(len(test_x)) :
        pre = clf.predict(test_x[i])
        if test_y[i] == pre :
            right += 1
    right_rate = right/len(test_x)
    return right_rate

if __name__ == "__main__":
    np.set_printoptions(threshold=np.nan) 
    img_file = "data_batch_1"
    dic = unpickle(img_file)
    train_num = 1000
    total_num = train_num+1000
    total_imgs = dic["data"][:total_num]
    total_labels = dic["labels"][:total_num]
    clusters =  100
    w = 16
    s = 8
    samples = 2000
    
    ''' to divide into patch '''

    X =  divide_patches(total_imgs,w,s)
    X = random.sample(X,samples)
#     Xnorm = random.sample(Xnorm,samples)
    ''' kMeans '''
    kmeans = KMeans(n_clusters = clusters)
    kmeans.fit(X)
     
    centroids = kmeans.cluster_centers_
    k_labels = kmeans.labels_ 
    print ("kmeans over")
    uz = (kmeans.inertia_)/samples
     
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
            for i in range(clusters):
                new_patch[i] = max([0,uz-euclDistance(patch, centroids[i,:])])
                new_img = np.concatenate((new_img,new_patch),axis = 0)
        new_imgs.append(new_img)
    print labels2
 
 
    ''' perceptron for predicting'''
    train = new_imgs[:train_num]
    train_label = total_labels[:train_num]
    clf = Perceptron(n_iter=100).fit(train, train_label)
    test_img  = new_imgs[train_num:total_num]
    test_labels = total_labels[train_num:total_num]
    print "Perceptron done"
    '''SVM for predicting'''
#     accuracyRate = my_svm(train, train_label, test_img, test_labels)  
#     print accuracyRate
    
    right = 0
    for i in range(len(test_img)):
        Z = clf.predict(test_img[i])
        if Z == test_labels[i]:
            right = right +1 
    print right/float(len(test_img))
endtime = datetime.datetime.now()
print (endtime - starttime).seconds   
    
                    

