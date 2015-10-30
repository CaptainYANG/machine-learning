'''
Created on Jan 21, 2015

@author: yuyin
'''

import codecs
import re
import numpy
import math
judgment = set()
judgeCont = []
user = {}
ratingTemp = []
eachMovie = []
movie = {}
movieTemp = {}
judge = {}
rating = []
judgePerUser = []
def mean(list):
    return float (sum(list))/len(list)

def var(list):
    a = mean(list)
    v = 0.0;
    for e in list:
        v += (e-a)**2
    return v/len(list)
def sd(list):
    a = var(list)
    return a**0.5
def sortSimi(list):
    simi = sorted(list.items(), key=lambda item: item[1],reverse = True)
    return simi
with codecs.open("movie_lens.csv","r",encoding = "utf8") as my_file:
    for line in my_file:
        judgment.add(line)
        line = line.strip()
        judgeCont = line.split("|")
        rating.append(int(judgeCont[2]))
        if judgeCont[1] != "unknown":
            eachMovie = re.split('[()]', judgeCont[1])
            for y in eachMovie:
                if y.isdigit():
                    movie[judgeCont[1]] = y
                    if judgeCont[0] in judge:
                        movieTemp = judge[judgeCont[0]]
                    else:
                        movieTemp = {}
                    movieTemp[judgeCont[1]] = [y, judgeCont[2]]
                    judge[judgeCont[0]] = movieTemp
                    break
        else :
            if judgeCont[0] in judge:
                movieTemp = judge[judgeCont[0]]
            else:
                movieTemp = {}
            movieTemp[judgeCont[1]] = ["unknown",judgeCont[2]]
            judge[judgeCont[0]] = movieTemp        
    items = sorted(movie.items(), key=lambda item: item[1],reverse = True)
for e in judge:
    judgePerUser.append(len(judge[e]))
print len(judgment)
print len(movie)
print items[0:10], items[len(items)-1]
print mean(rating),sd(rating)
print mean(judgePerUser),sd(judgePerUser)
print min(judgePerUser), max(judgePerUser),sum(judgePerUser)  
def similarity(a,b):
    simiAUser = []
    simiBUser = []
    for e in judge:
        if a in judge[e]:
            if b in judge[e]:
                simiAUser.append(int(judge[e][a][1]))
                simiBUser.append(int(judge[e][b][1]))
    if len(simiAUser) >= 4: 
        simil = numpy.corrcoef(simiAUser, simiBUser)[0, 1]
        if math.isnan(simil):
            return 0
        else:
            return simil
print similarity(u'Scream (1996)', u'Stargate (1994)')
def allMovieSimi(a):
    allSimi = {}
    for e in movie:
        if e !=  a:
            allSimi[e] = similarity(a, e)
    return sorted(allSimi.items(), key=lambda item: item[1],reverse = True)
list =  allMovieSimi(u'Scream (1996)')
print list[0:5]
list = allMovieSimi(u'Stargate (1994)')
print list[0:5]
def similarityZero(a,b):
    simiAUser = []
    simiBUser = []
    for e in judge:
        if a in judge[e]:
            simiAUser.append(int(judge[e][a][1]))
            if b in judge[e]:
                simiBUser.append(int(judge[e][b][1]))
            else:
                simiBUser.append(0)
        elif b in judge[e]:
            simiAUser.append(0)
            simiBUser.append(int(judge[e][b][1]))
    if len(simiAUser) >= 4: 
        simil = numpy.corrcoef(simiAUser, simiBUser)[0, 1]
        if math.isnan(simil):
            return 0
        else:
            return simil
def allMovieSimiZero(a):
    allSimi = {}
    for e in movie:
        if e !=  a:
            allSimi[e] = similarityZero(a, e)
    return sorted(allSimi.items(), key=lambda item: item[1],reverse = True)
list =  allMovieSimiZero(u'Scream (1996)')
print list[0:5]
list = allMovieSimiZero(u'Stargate (1994)')
print list[0:5]
