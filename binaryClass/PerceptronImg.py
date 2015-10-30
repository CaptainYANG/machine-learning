import cPickle, gzip, numpy
import matplotlib.pyplot as plt
from scipy.linalg import norm
from scipy import sum, average
from numpy import array, dot, random

random.seed(21)

# def print_img(im):
#     lm = im.reshape(28,28)
#     plt.show()
#     plt.imshow(lm.reshape(28,28) , plt.cm.gray)




def build_set(training_data,train_set,num1):
    i=0
    for img in train_set[0]:
        label = train_set[1][i]
        #print label
        #print_img(img)
        if (label == num1):
            img = numpy.append([1.],img) # add costant
            training_data.append((img,1,label))
        elif (label != num1):
            img = numpy.append([1.],img) # add costant
            training_data.append((img,-1,label))
 

        i+=1
        



# def test_img(mean,not_mean):
#     i=0
#     for img in train_set[0]:
#         label = train_set[1][i] # label
#        
#         diff1 = compare_images(mean,img)[0]
#         diff2 = compare_images(not_mean,img)[0]
#         if diff1<=diff2 :
#             print label
#         i+=1


def func(x):
    if x>=0:
        return 1
    else:
        return -1
     


def perceptron(epoch, total):

   # unit_step = lambda x: 0 if x < -1 else 1

    w = random.rand(785)
    errors = []
    eta = 0.5
    correct = 0
    globalError = 0
    learned = False
    iteration = 0
    while not learned:
        globalError = 0.0
        correct = 0
        items = 0

        for x, expected, lbl in training_data:
            items += 1
            #index = random.randint(0,len(training_data))
            #(x, expected, lbl) = training_data[index]
            result = dot(w,x)
            if expected != func(result):
                error = expected - func(result)
                w += eta * error * x
                globalError += abs(error)
            else:
                correct+=1
            #print items
            if items == total:
                break
        errors.append(globalError)
        iteration += 1
        if correct >= total or iteration >= epoch: # stop criteria
            print 'iterations',epoch
            #print 'Global error: ',globalError
            learned = True # stop learning
   

    return (total - correct)/float(total)
    

 
def test(test_data,w):

    #unit_step = lambda x: 0 if x < -1 else 1

    right = 0
    wrong = 0
    for x,expected,label in test_data:
        result = dot(x, w)
        
        if func(result)!=expected:
           # print("{}: {} -> {} {}".format(label, result, func(result),clas))
            wrong +=1
        else:
            right +=1
    #print right
    #print wrong
    print "Sensitivity=", right/float(right+wrong)

     
# Load the dataset

f = gzip.open('mnist.pkl.gz', 'rb')
train_set, valid_set, test_set = cPickle.load(f)
f.close()

w = []
"""
for i in range(10):
    training_data = []
    build_set(training_data,train_set,7)

    w.append(perceptron())
    print "End number: "
"""
training_data = []
build_set(training_data,train_set,5)
errors = []
for epoch in range(1,101):
    error_now = perceptron(epoch,10000)
    errors.append(error_now)
    if error_now == 0.0:
        break
plt.plot(errors)
plt.show()

#test_data = []
#build_set(test_data,test_set,7)

#test(test_data,w)

#test_img(class7,class_not7)
print "END"
"""
n_m, n_0 = compare_images(train_set[0][15], train_set[0][29])
print "Manhattan norm:", n_m, "/ per pixel:", n_m/train_set[0][15].size
print "Zero norm:", n_0, "/ per pixel:", n_0*1.0/train_set[0][15].size


im = train_set[0][0] # la premire image
im = numpy.append([1.],im) # ajoute une composante
label = train_set[1][0] # son label
#numpy.dot(w,im) # le produit scalaire avec un vecteur de  taille

"""

