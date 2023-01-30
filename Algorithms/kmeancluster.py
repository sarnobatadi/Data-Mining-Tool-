import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
import seaborn as sns
from sklearn.cluster import KMeans 
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import MinMaxScaler
import json
import math
import copy
from random import shuffle
import pandas as pd


def read_data_from_csv(filename):
    f = open(filename)
    lines = f.readlines()
    data = {}
    headers = lines[0].split(',')
    points = []
    for  i in range(1,len(lines)):
        ordinates = list(map(float,lines[i].split(',')))
        point = {}
        for j in range(len(headers)):
            point[headers[j]]=ordinates[j]
        points.append(point)
    data['ordinates'] =headers
    data['points']=points
    return data


# In[3]:

def euclidean_distance(a,b,ordinates):
    distance=0
    for i in range(len(ordinates)):
        distance+=(a[ordinates[i]]-b[ordinates[i]])**2
    return distance**0.5


# In[4]:

def update_centroid(cluster,ordinates):
    for ordinate in ordinates:
        temp = 0
        for point in cluster['points']:
            temp+=point[ordinate]
        cluster['centroid'][ordinate]=(temp/len(cluster['points']))
    return cluster


# In[5]:

def add_points_into_clusters(clusters,points,ordinates):
    for cluster in clusters:
        cluster['points']=[]
    
    for point in points:
        min_distance = euclidean_distance(point,clusters[0]['centroid'],ordinates)
        min_index = 0
        for i in range(1,len(clusters)):
            temp = euclidean_distance(point,clusters[i]['centroid'],ordinates)
            if temp<min_distance:
                min_distance=temp
                min_index=i
        clusters[min_index]['points'].append(point)
        clusters[min_index]=update_centroid(clusters[min_index],ordinates)

    return clusters


# In[6]:

def kmeans_clustering(data,k,iterations):
    clusters = []
    points = data['points']
    ordinates = data['ordinates']
    if k >=len(points):
        for point in points:
            clusters[i].append(point)
        return clusters    
    shuffle(points)
    for i in range(k):
        cluster = {
            'centroid':0,
            'points':[]
        }
        cluster['centroid'] = copy.deepcopy(points[i])
        clusters.append(cluster)
        
    iteration=0
    while iteration<iterations:
        clusters = add_points_into_clusters(clusters,points,ordinates)
        iteration+=1
        # for i in range(0,len(clusters)):
        #     print('Cluster No: ',i+1)
        #     print('\tCentroid:',str(clusters[i]['centroid']))
        #     print('\tPoints:',str(clusters[i]['points']))
       
    return clusters



def print_clusters(clusters):
    res = ""
    for i in range(0,len(clusters)):
        print('Cluster No: ',i+1)
        res +='\nCluster No: '+ str(i+1)
        print('\tCentroid:',str(clusters[i]['centroid']))
        res +='\tCentroid:' + str(clusters[i]['centroid'])
        # print('\tPoints:',str(clusters[i]['points']))
    return res

# In[8]:


def kmeanscluster(clust,itr):
    iris = pd.read_csv("iris.csv")
    x = iris.iloc[:, [ 1, 2, 3]].values

    iris_setosa=iris.loc[iris["Species"]=="setosa"]
    iris_virginica=iris.loc[iris["Species"]=="virginica"]
    iris_versicolor=iris.loc[iris["Species"]=="versicolor"]

    kmeans = KMeans(n_clusters = clust, init = 'k-means++', max_iter = itr, n_init = 10, random_state = 0)
    y_kmeans = kmeans.fit_predict(x)
    plt.scatter(x[y_kmeans == 0, 0], x[y_kmeans == 0, 1], s = 100, c = 'purple', label = 'Iris-setosa')
    plt.scatter(x[y_kmeans == 1, 0], x[y_kmeans == 1, 1], s = 100, c = 'orange', label = 'Iris-versicolour')
    plt.scatter(x[y_kmeans == 2, 0], x[y_kmeans == 2, 1], s = 100, c = 'green', label = 'Iris-virginica')
    print(x[y_kmeans == 2, 0])
    #Plotting the centroids of the clusters
    plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:,1], s = 100, c = 'red', label = 'Centroids')
    plt.title("K-means Visualization")
    # plt.xlim(0, 60)
    plt.xlabel("SepalLengthCm")
    plt.ylabel("PetalLengthCm")
    # plt.legend()
    plt.show()
    c = kmeans_clustering(read_data_from_csv('kmeans.csv'),clust,itr)
    return(print_clusters(c))

# kmeanscluster(3,300)