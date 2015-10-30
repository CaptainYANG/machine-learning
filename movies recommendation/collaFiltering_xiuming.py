import codecs
import re
import sys
from numpy import power, sqrt, loadtxt, average
import numpy as np
import matplotlib.pyplot as plt
from numpy.matlib import randn
from array import array
import math



#import colections
#d = defauldict(dict)

# Extract the total number of judgment, the total number of different user, 
#must inside the with as structure
def open_file(file_name):
    with codecs.open(file_name, "r", encoding = "utf-8") as my_file:
        return my_file

file_name = "Movie.csv"
file_name = "movie_lens.csv"

def basic_info(f):
    with codecs.open(f, "r", encoding = "utf-8") as my_file:
        info = []
        for line in my_file:
            line = line.strip()
#             line = line.split('|')
            info += [line.split('|'),]
        return info

file =  basic_info(file_name)
# print basic_info(file_name)

sum = set()
for i in file:
    am = i[0] +i[1]+i[2]
    sum.add(am)
print len(sum)

def count_diff_judgement(f):
    return len(basic_info(f))

print count_diff_judgement(file_name)

def count_diff_users(f):
    users = set()
    for el in basic_info(f):
        users.add(el[0])
    return len(users)

# print "users %s" %count_diff_users(file_name)

def count_diff_movies(f):
    movies = set()
    for el in basic_info(f):
        movies.add(el[1])
    return len(movies)

# print count_diff_movies(file_name)
#------the most recent movie-----#

def partition(list):
    pivot = list[len(list)-1]
    left = []
    right = []
    for p in range(0, len(list)-1):
        el = list[p]
        if el > pivot:
            left += [el]
        else:
            right += [el]

    return (left, pivot, right)



    
# findLongest()
def quickSort(list):
    length = len(list)
    if(length <= 1):
        return list
    (left, pivot, right) = partition(list)
    sortedLeft =  quickSort(left)
    sortedRight = quickSort(right)
    return sortedLeft + [pivot] + sortedRight


def getByRegex(content, reg):
    pattern = re.compile(reg)
    match = pattern.search(content)
    if match:
        return match.group()
    else:
        return 0
def build_movie_sortedlist():      
    movie_info={}
    for el in basic_info(file_name):
        try:
            value = el[1]
            regex =  r'\('+r'[\d]{4}'+r'\)'
            pattern = re.compile(regex)
            match = pattern.search(value)
            year =  match.group()
            name = value[:-7]
    #         print year,name
        except:
    #         print value
    #         print sys.exc_info()
            continue
    
            
    #         name = el[1].split("(")[0]
    #         year = el[1].split("(")[1].strip(")")
    #         print name, year
    
        if year not in movie_info:
            movie_info[year] =  set()
            movie_info[year].add(name)
        else:
            movie_info[year].add(name)
    #TODO dic->list        
    movie_list = []
    for el in movie_info:
        year_movies =  {el: movie_info[el]}
        movie_list.append(year_movies)
    
    
    movie_sorted_list = quickSort(movie_list)
    length = len(movie_sorted_list)
    # print movie_sorted_list[0]
    # print movie_sorted_list[length-1]
    return movie_sorted_list

# list =  build_movie_sortedlist()
# print list[0]
  

#----------problem 3---------------#
def count_mean(f):
    sum =  0.0
    num =  len(f)
    for el in f:
        sum += float(el[2])
    return sum/num
print count_mean(file)

def count_standard_deviation(f):
#     num =  len(f)
#     eq1 = 0.0
#     for el in f:
#         r = float(el[2])
#         r_pow = r * r
#         eq1 += r_pow
#     eq1 = eq1/(num-1)
#     eq2 = power(count_mean(f),2)
    ratings = []
    for e in f:
        ratings.append(int(e[2]))
#     print "rating mean %s" %np.average(ratings)
#     print ratings
    return np.std(ratings)
count_standard_deviation(file)
# print count_mean(file)
# print count_standard_deviation(file)
def show_distribution(file):
#     ratings = {0:0,1:0,2:0,3:0,4:0,5:0}
    z = []
    for i in file:
        z += [int(i[2]),]
    z = np.array(z)
    n,bins,patches = plt.hist(z,5,facecolor='y',alpha=0.75)
    plt.xlabel('Rating')
    plt.ylabel('Quantity')
    plt.title('Histogram of Rating')

    plt.text(1,30000,r'$\mu=3.5299,\ \sigma=1.1257$')
    plt.axis([0,5,0,35000])
 
#     plt.axis([40,160,0,0.03])
    plt.grid(True)
#     plt.hist(z,5,)
    plt.show()
show_distribution(file)
#----------Problem 5----------#
def judges_basic_info(file):
    user_judges = {}
    for el in file:
        user = el[0]
        if user in user_judges:
            user_judges[user]+=1
        else:
            user_judges[user]=1
    judges = []
#     print user_judges
    for user in user_judges:
        judges += [user_judges[user],]
    # print judges
    judges = np.array(judges)
#     print "sum %s" %np.sum(judges)
    avg_judge = np.average(judges)
    print avg_judge
    std_dev_judge = np.std(judges,axis=0)
    print std_dev_judge  
    print np.max(judges,axis=0)
    print np.min(judges,axis=0)
    
judges_basic_info(file)
    
# def recommendation(file):
movies = set()
movies_dic = {}
for e in file:
    movie = e[1]
    user = e[0]
    rating = int(e[2])
    if movie in movies:
        movies_dic[movie][user] = rating
    else:
        movies.add(movie)
        movies_dic[movie] = {}
        movies_dic[movie][user] = rating

#filter the common users reviewing the two movies:
def filter_common_users(name1, name2, movies_dic):
    users1 = movies_dic[name1]
    users2 = movies_dic[name2]
    movie1_rating = []
    movie2_rating = []
#     print len(users2)
    for e in users1:
        if e in users2:
            movie1_rating.append(users1[e])
            movie2_rating.append(users2[e])
#             print movie1_rating
#             print movie2_rating
    return (movie1_rating,movie2_rating)
    
name1 = u'Scream (1996)' 
name2 = u'Stargate (1994)'

name1 = u'Scream (1996)' 
# name2 = u'Turbo: A Power Rangers Movie (1997)'
(a,b) = filter_common_users(name1, name2, movies_dic)

def calculate_correlation(a,b):
    return np.corrcoef(a, b)[0,1]
#     return np.dot(a,b)/(math.sqrt(np.dot(a,a))*math.sqrt(np.dot(b,b)))

# print a
print b
print "here %s" %calculate_correlation(a, b)

#--------- problem 8-----#
# name1 = u'Stargate (1994)'
name1 = u'Scream (1996)'
films_set = set()
for e in file:
    films_set.add(e[1])
     
film_correlation = {}
films = films_set.copy()
 
films.remove(name1)
for f in films:
    name2 = f
    (a,b) = filter_common_users(name1, name2, movies_dic)
    try:
        if len(a) > 7:
            result = calculate_correlation(a, b)
            film_correlation[name2] = result
#             print name2
#             print a
#             print b
#             print film_correlation[name2]
    except:
#         print a
#         print b
        break
    

(a,b) = filter_common_users(name1, name2, movies_dic)

film_rating = []
for e in film_correlation:
    rate = film_correlation[e]
    if math.isnan(rate):
        pass
    else:
        film_rating.append((rate, e))


q = quickSort(film_rating)
# print q[-1]
print "!!!!"
for e in q[0:5]:
    print e


name2 = u'Leading Man, The (1996)'
(a,b) = filter_common_users(name1, name2, movies_dic)
# print a
# print b
# print film_correlation[:5]
# print film_correlation
# (a,b) = filter_common_users(u'R', u'To Cross the Rubicon (1991)', movies_dic)
# print a
# print b
# print calculate_correlation(a, b)
#----------Problem 11-------#
# users=  set()
# for e in file:
#     users.add(e[0])
# for movie in movies_dic:
#     for u in users:
#         if u not in movies_dic[movie]:
#             movies_dic[movie][u] = 0
# 
# film_correlation = {}
# name1 = u'Stargate (1994)'
# for movie in movies_dic:
#     user_dic = movies_dic[movie]
#     user_list = []
#     for u in users:
#         user_list.append(user_dic[u])
#     movies_dic[movie] = user_list
# 
# for name2 in movies:
#     result =  calculate_correlation(movies_dic[name1], movies_dic[name2])
#     if math.isnan(result):
#         pass
#     else:
#         film_correlation[name2] = result

#yishang  unknown problem

users=  set()
for e in file:
    users.add(e[0])
movieset = set()
for e in file:
    if e[1] != 'unknown':
        movieset.add(e[1])
movies = {}
for movie in movieset:
    movies[movie] = {}
    for user in users:
        movies[movie][user] = 0
# print movies

for e in file:
    user = e[0]
    rating = int(e[2])
    movie = e[1]
    if movie != 'unknown':
        movies[movie][user] = rating
    
# print movies
film_correlation = {}
name1 = u'Scream (1996)'
for movie in movieset:
    user_dic = movies[movie]
    user_list = []
    for u in users:
        user_list.append(user_dic[u])
    movies[movie] = user_list
 
for name2 in movies:
    result =  calculate_correlation(movies[name1], movies[name2])
    if math.isnan(result):
        pass
    else:
        film_correlation[name2] = result
list = sorted(film_correlation.items(), key = lambda film_correlation:film_correlation[1],reverse = True)
for e in list[1:6]:
    print e
        
    
    
