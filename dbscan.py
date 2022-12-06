from logging import PlaceHolder
import math
from multiprocessing import Value
import pylab as pl
import numpy as np
import numpy.random as random
from numpy.core.fromnumeric import *
from sklearn import datasets
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.pyplot as plt
import math as m
from sklearn.datasets import make_blobs
import plotly.figure_factory as ff
from scipy.cluster import hierarchy
from scipy.cluster.hierarchy import dendrogram, linkage
from random import randint
import pandas as pd


dataset = []
def calDist(X1, X2):
    sum = 0
    for x1, x2 in zip(X1, X2):
        sum += (x1 - x2) ** 2
    return sum ** 0.5
    return (((X1[0]-X2[0])**2)+(X1[1]-X2[1])**2)**0.5

def getNeibor(data, dataSet, e):
    res = []
    for i in range(len(dataSet)):
        if calDist(data, dataSet[i]) < e:
            res.append(i)
    return res

def DBSCAN(dataSet, e, minPts):
    coreObjs = {}
    C = {}
    
    n = dataset
    for i in range(len(dataSet)):
        neibor = getNeibor(dataSet[i], dataSet, e)
        if len(neibor) >= minPts:
            coreObjs[i] = neibor
    oldCoreObjs = coreObjs.copy()
    # st.write(oldCoreObjs)
    # CoreObjs set of COres points
    k = 0
    notAccess = list(range(len(dataset)))

    # his will check the relation of core point with each other
    while len(coreObjs) > 0:
        OldNotAccess = []
        OldNotAccess.extend(notAccess)
        cores = coreObjs.keys()
        randNum = random.randint(0, len(cores))
        cores = list(cores)
        core = cores[randNum]
        queue = []
        queue.append(core)
        notAccess.remove(core)
        while len(queue) > 0:
            q = queue[0]
            del queue[0]
            if q in oldCoreObjs.keys():
                delte = [val for val in oldCoreObjs[q]
                            if val in notAccess]
                queue.extend(delte)
                notAccess = [
                    val for val in notAccess if val not in delte]
        k += 1
        C[k] = [val for val in OldNotAccess if val not in notAccess]
        for x in C[k]:
            if x in coreObjs.keys():
                del coreObjs[x]

    print(C)
    return C

def draw(C, dataSet,a1,a2):
    color = ['r', 'y', 'g', 'b', 'c', 'k', 'm']
    vis = set()
    for i in C.keys():
        X = []
        Y = []
        datas = C[i]
        for k in datas:
            vis.add(k)
        for j in range(len(datas)):
            X.append(dataSet[datas[j]][0])
            Y.append(dataSet[datas[j]][1])
        plt.scatter(X, Y, marker='o',
                    color=color[i % len(color)], label=i)
    vis = list(vis)
    unvis1 = []
    unvis2 = []
    for i in range(len(dataSet)):
        if i not in vis:
            unvis1.append(dataSet[i][0])
            unvis2.append(dataSet[i][1])
    #st.subheader("Plot of cluster's after DBSCAN ")
    plt.xlabel(a1)
    plt.ylabel(a2)
    plt.scatter(unvis1, unvis2, marker='o', color='black')
    plt.legend(loc='lower right')
    plt.show()
    #st.pyplot()

def dbscanres(eps,mp):
    cols = []
    data = pd.read_csv('iris.csv')
    for i in data.columns[:-1]:
        cols.append(i)
    # atr1, atr2 = st.columns(2)
    # attribute1 = st.selectbox("Select Attribute 1", cols)
    # attribute2 = st.selectbox("Select Attribute 2", cols)
    
    arr1 = []
    arr2 = []
    attribute1 = "Sepal.Width"
    attribute2 = "Sepal.Length"
    for i in range(len(data)):
        arr1.append(data.loc[i, attribute1])
    for i in range(len(data)):
        arr2.append(data.loc[i, attribute2])
    for i in range(len(arr1)):
        tmp = []
        tmp.append(arr1[i])
        tmp.append(arr2[i])
        dataset.append(tmp)
    # r = st.number_input('Insert value for eps', value=0.09)
    # mnp = st.number_input(
    #     'Insert mimimum number of points in cluster', step=1, value=7)
    C = DBSCAN(dataset, 0.25,12)
    draw(C, dataset,attribute1,attribute2)
    res = "Clusters : "
    res +=  "\n" + str(C)
    return res

# dbscanres(0.25,12)
