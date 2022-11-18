import random

def kmeansClustering(k,datalist):
    clusters = []
    flg = 0
    for i in range(2):
            cluster = {
            'centroid':0,
            'dist':[],
            'clust':[]
            }
            cluster['centroid'] = random.randint(datalist[0], datalist[len(datalist)-1])
            clusters.append(cluster)
    k = clusters[0]
    k['dist'] = []
    k = clusters[1]
    k['dist'] = []
    for i in datalist:
        for k in clusters:
            k['dist'].append(abs(k['centroid']-i))
    
    k = clusters[0]
    k['clust']= []
    k = clusters[1]
    k['clust']= []
    for i in range(len(datalist)-1):
        dist1 = clusters[0]['dist'][i]
        dist2 = clusters[1]['dist'][i]
        if(dist1<dist2):
            k = clusters[0]
            k['clust'].append(datalist[i])
        else:
            k = clusters[1]
            k['clust'].append(datalist[i])

    print(clusters)
    while(flg==0):
        k = clusters[0]
        k['dist'] = []
        k = clusters[1]
        k['dist'] = []
        for i in datalist:
            for k in clusters:
                k['dist'].append(abs(k['centroid']-i))
        
        k = clusters[0]
        k['clust']= []
        k = clusters[1]
        k['clust']= []
        for i in range(len(datalist)-1):
            dist1 = clusters[0]['dist'][i]
            dist2 = clusters[1]['dist'][i]
            if(dist1<dist2):
                k = clusters[0]
                k['clust'].append(datalist[i])
            else:
                k = clusters[1]
                k['clust'].append(datalist[i])

        for k in clusters:
            print(k['clust'])
            centroid = 0
            if(len(k['clust'])>0):
                centroid = sum(k['clust'])/len(k['clust'])
            if(k['centroid'] == centroid):
                flg = 1
            k['centroid'] = centroid
            
        # if(flg == 1):
        #     break
        print(clusters)
        # flg = 1
    
        



datalist = [2,4,10,12,3,20,30,11,25]
kmeansClustering(2,datalist)