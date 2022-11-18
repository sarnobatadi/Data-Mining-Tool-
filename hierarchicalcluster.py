from sklearn import datasets
from scipy.cluster.hierarchy import dendrogram
import matplotlib.pyplot as plt
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import adjusted_rand_score
from sklearn import preprocessing
from scipy.cluster.hierarchy import linkage

def find_row_col_of_min_from_2d(data):
    mat = data['mat']
    min_element = min(mat[0])
    min_col = mat[0].index(min_element)
    min_row = 0
    for i in range(1,len(mat)):
        min_in_row = min(mat[i][0:i])
        if min_in_row <min_element or min_element==0:
            min_col = mat[i].index(min_in_row)
            min_row = i
            min_element=min_in_row
            
    return min_row,min_col


def merge_clusters_and_update_mat(data,min_location):
    new_mat = []    
    
    min_index_of_location = min(min_location)
    max_index_of_location = max(min_location)
    data['cols'][min_index_of_location]+=data['cols'][max_index_of_location]
    data['rows'][min_index_of_location]+=data['rows'][max_index_of_location]
    del data['cols'][max_index_of_location]
    del data['rows'][max_index_of_location]
    mat = data['mat']
    for i in range(1,len(mat)):
        new_row = []
        for j in range(1,len(mat)):
            value = mat[i-1][j-1]
            if i-1==min_index_of_location:
                value=min(value,mat[max_index_of_location][j-1])
            elif j-1==min_index_of_location:
                value=min(value,mat[max_index_of_location][i-1])                
            new_row.append(value)
            
        new_mat.append(new_row)    
    data['mat']=new_mat
    return data


def clustering(data):
    while len(data['cols'])!=1:
        print_2d_mat(data,False)
        min_location = find_row_col_of_min_from_2d(data)
        data = merge_clusters_and_update_mat(data,min_location)
    print_2d_mat(data,False)



def hcluster(clust):
    iris = datasets.load_iris()
    ward = AgglomerativeClustering(n_clusters=3)
    ward_pred = ward.fit_predict(iris.data)

    complete = AgglomerativeClustering(n_clusters=3, linkage='complete')

    complete_pred = complete.fit_predict(iris.data)

    avg = AgglomerativeClustering(n_clusters=3, linkage='average')

    avg_pred = avg.fit_predict(iris.data)


    ward_ar_score = adjusted_rand_score(iris.target, ward_pred)
    complete_ar_score = adjusted_rand_score(iris.target, complete_pred)
    avg_ar_score = adjusted_rand_score(iris.target, avg_pred)

    normalized_X = preprocessing.normalize(iris.data)
    ward = AgglomerativeClustering(n_clusters=3)
    ward_pred = ward.fit_predict(normalized_X)

    complete = AgglomerativeClustering(n_clusters=3, linkage="complete")
    complete_pred = complete.fit_predict(normalized_X)
    avg = AgglomerativeClustering(n_clusters=3, linkage="average")
    avg_pred = avg.fit_predict(normalized_X)

    ward_ar_score = adjusted_rand_score(iris.target, ward_pred)
    complete_ar_score = adjusted_rand_score(iris.target, complete_pred)
    avg_ar_score = adjusted_rand_score(iris.target, avg_pred)
    res = ""
    print( "Scores: \nWard:", ward_ar_score,"\nComplete: ", complete_ar_score, "\nAverage: ", avg_ar_score)
    res += "Scores: \nWard:" + str(ward_ar_score) + "\nComplete: " + str(complete_ar_score) + "\nAverage: " + str(avg_ar_score)
    # Import scipy's linkage function to conduct the clustering
    linkage_type = 'ward'
    linkage_matrix = linkage(normalized_X, linkage_type)
    plt.figure(figsize=(22,18))
    dendrogram(linkage_matrix)
    plt.show()
    return res

# hcluster(3)