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
def normalize(vector):
    if np.std(vector) == 0 :
        new_vector = [0 for i in range(len(vector))]
    else :
        new_vector = (vector - np.mean(vector))/np.std(vector)
    return new_vector

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
                patch =  patch.reshape(w * w * 3)
                patches.append(patch)
            cur_x = 0
            cur_y += stride
    return patches

# def divide_patches(train_imgs):
#     after_patches = []
#     for img in train_imgs:
#         patches = [[]for i in range(8)]
#         img = img.reshape(3,32,32)
#         rdr_list = range(26)
#         sele_r = random.sample(rdr_list,8)
#         sele_c = random.sample(rdr_list,8)
#         for i in range(8):
#             patches[i] = img[:,sele_r[i]:sele_r[i]+6,sele_c[i]:sele_c[i]+6].reshape(1,108)
#             patches[i] = np.array(normalize(patches[i]))
#             after_patches.append(patches[i])
#     print after_patches[1]
#     return after_patches

if __name__ == "__main__":
    np.set_printoptions(threshold=np.nan) 
    img_file = "data_batch_1"
    dic = unpickle(img_file)

    total_num = 200
    total_imgs = dic["data"][:total_num]
    total_labels = dic["labels"][:total_num]
    
#     imgs = np.array([])
    
    #TODO to divide into patch 
    X =  divide_patches(total_imgs,16,16)
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