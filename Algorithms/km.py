
# coding: utf-8

# In[1]:
import matplotlib.pyplot as plt
import json
import math
import copy
from random import shuffle
import pandas as pd

# In[2]:

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
    for i in range(0,len(clusters)):
        print('Cluster No: ',i+1)
        print('\tCentroid:',str(clusters[i]['centroid']))
        print('\tPoints:',str(clusters[i]['points']))


# In[8]:

c = kmeans_clustering(read_data_from_csv('kmeans.csv'),3,5)
fig, ax = plt.subplots(figsize=(4, 4))
# print(c[0]['centroid']['"Sepal.Length"'])
centroids_x = [c[x]['centroid']['Sepal.Length'] for x in range(len(c[1]))] #SepalLengthCm: [0] 
centroids_y = [c[x]['centroid']['Sepal.Width'] for x in range(len(c[1]))] #PetalLengthCm: [2]
print(centroids_x)
print(centroids_y)
data = pd.read_csv('kmeans.csv')
x = data['Sepal.Length']
y = data['Sepal.Width'] 
print(x)
print(y)
assignments = c[0]
plt.scatter(x, y)
plt.plot(centroids_x,centroids_y, c='white', marker='.', linewidth='0.01', markerfacecolor='red', markersize=22)
plt.title("K-means Visualization")
plt.xlabel("SepalLengthCm")
plt.ylabel("PetalLengthCm")
plt.show()


