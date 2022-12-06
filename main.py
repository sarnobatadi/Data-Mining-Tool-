import eel
from sklearn.preprocessing import StandardScaler
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from collections import Counter
from random import randint
import math
import itertools
from tkinter import ttk
import pylab as py
import statsmodels.api as sm
import matplotlib.pyplot as plt
import pandas as pd
import math
import scipy.stats as stats
import pylab 
import plotly.express as px
from scipy.stats import chi2_contingency
from scipy.stats import chi2
from numpy.random import randn
from numpy.random import seed
from hierarchicalcluster import hcluster
from scipy.stats import pearsonr
from sklearn import tree
import numpy as np
import os
from tkinter import *
from tkinter import filedialog
import seaborn as sns
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from ann import basicANN
from kmedoid import kmcall
from kmeancluster import kmeanscluster
from pagerank import pgrank_res,HIT_res
from crawler import bfscrawl,dfscrawl
from dbscan import dbscanres
eel.init("web")

# df = pd.read_excel('iris.xlsx')

@eel.expose
def dbres(ep,mp):
    return dbscanres(float(ep),float(mp))

@eel.expose
def crawl(url,type):
    if(type=="bfs"):
        return bfscrawl(url)
    else:
        return dfscrawl(url)

@eel.expose
def pgrank(itr,dump):
    return pgrank_res(int(itr),float(dump))

@eel.expose
def hits(itr):
    return HIT_res(int(itr))


def findMean(col):
	global df 
	y = df[col]
	n = len(y) 
	get_sum = sum(y) 
	mean = get_sum / n 
	ans="Mean is: " + str(mean)
	return ans


def findMedian(col):
    global df 
    y = df[col]
    n = len(y) 
    newy=sorted(y)
    if n % 2 == 0: 
        median1 = newy[n//2] 
        median2 = newy[n//2 - 1] 
        median = (median1 + median2)/2
    else: 
        median = newy[n//2] 
    ans="Median is: " + str(median)
    return ans

def findMode(col):
    global df 
    y = df[col]
    n = len(y) 
        
    data = Counter(y) 
    get_mode = dict(data) 
    mode = [k for k, v in get_mode.items() if v == max(list(data.values()))] 
        
    if len(mode) == n: 
        get_mode = "No mode found"
    else: 
        get_mode = "Mode is / are: " + ', '.join(map(str, mode)) 
    print(get_mode)
    return get_mode

def findMidrange(col):
    global df 
    y = df[col]
    n = len(y) 
    data=sorted(y) 
    maxV=data[n-1]
    minV=data[0]
    ans_midrange=(minV+maxV)/2
    ans="Midrange: "+str(ans_midrange)
    print(ans)
    return ans

def findvariance(col):
    global df 
    y = df[col]
    n = len(y)
    sum_y=sum(y)
    mean =sum_y/n
    deviations = [(x - mean) ** 2 for x in y]
    variance = sum(deviations) / n
    ans="Variance: "+str(variance)
    print(ans)
    return ans

def findstddeviation(col):
    global df 
    y = df[col]
    ddof=0
    n = len(y)
    mean = sum(y) / n
    var= sum((x - mean) ** 2 for x in y) / (n - ddof)
    std_dev = math.sqrt(var)
    ans="Standard Deviation: "+str(std_dev)
    print(ans)
    return ans

def findrange(col):
    global df 
    y = df[col]
    n=len(y)
    data=sorted(y) 
    maxV=data[n-1]
    minV=data[0]
    ans_range=maxV-minV
    ans="Range: "+str(ans_range)
    print(ans)
    return ans

def calc_quantile(col):
    global df 
    y = df[col]
    y = sorted(y)
    ans=[]
    res="Quartiles are: "
    i=0.25
    res += "\nLower quartile(Q1) is  "+str(y[int((len(y)+1)/4)])
    res += "\nMiddle quartile(Q2) is "+str(y[int((len(y)+1)/2)])
    res +="\nUpper quartile(Q3) is "+str(y[int(3*(len(y)+1)/4)])
    return res

def calc_quantile_range(col):
    global df 
    y = df[col]
    y = sorted(y)
    ans="Interquantile Range: "+str(y[int(3*(len(y)+1)/4)]-y[int((len(y)+1)/4)])
    print(ans)
    return ans

def findfive_number_summary(col):
    global df 
    y = df[col]
    n = len(y) 
    newy=sorted(y)
    if n % 2 == 0: 
        median1 = newy[n//2] 
        median2 = newy[n//2 - 1] 
        median = (median1 + median2)/2
    else: 
        median = newy[n//2] 
    med_ans="Median is: " + str(median)
    s_lst = sorted(y)
    sum_ans=[]
    i=0.25
    for j in range(0,4):
        idx = (len(s_lst) - 1)*i
        int_idx = int(idx)
        remainder = idx % 1
        if remainder > 0:
            lower_val = s_lst[int_idx]
            upper_val = s_lst[int_idx + 1]
            sum_ans.append(lower_val * (1 - remainder) + upper_val * remainder)
        else:
            sum_ans.append(s_lst[int_idx])
        i=i+0.25
    res=med_ans+"\n1st Quartile: "+str(sum_ans[0])+"\n3rd Quartile: "+str(sum_ans[2])+"\nMinimum Element: "+str(min(y))+"\nMaximum Element: "+str(max(y))
    print("Five Number Summary:")
    print(res)
    return res


def qq_plot(col):
    global df 
    y = df[col]
    s_lst = sorted(y)
    ans=[]
    res="Quantiles are: "
    i=0.25
    for j in range(0,4):
        idx = (len(s_lst) - 1)*i
        int_idx = int(idx)
        remainder = idx % 1
        if remainder > 0:
            lower_val = s_lst[int_idx]
            upper_val = s_lst[int_idx + 1]
            ans.append(lower_val * (1 - remainder) + upper_val * remainder)
        else:
            ans.append(s_lst[int_idx])
        i=i+0.25
    stats.probplot(ans, dist="norm", plot=pylab)
    pylab.show()

def histogram(col):
    global df 
    y = df[col]
    plt.hist(y)
    plt.show()

def scatter_plot(col):
    global df 
    y = df[col]
    plt.scatter(y, y)
    plt.show()

def box_plot(col):
    global df 
    y = df[col]
    plt.boxplot(y)
    plt.show()

def chi2test_with_x_y(col1,col2,classN):
    global df
    attr1 = col1
    attr2 = col2
    category = classN
    arrClass = df[category].unique()
    group = df.groupby(category)
    f = {
    attr1: 'sum',
    attr2: 'sum'
    }
    res =""
    v1 = group.agg(f)
    print(v1)
    res += str(v1) + " "
    v = v1.transpose()
    print(v)
    res += "\n"
    for column in list(v.columns):
        res+=str(column)+" "
    res += "\n"
    df_rows = v.to_numpy().tolist()
    for row in df_rows:
        res+= str(row) + " "
        
    res += "\n"
    total = v1[attr1].sum()+v1[attr2].sum()
    chiSquare = 0.0
    #chisqure formula (observed - expected)^2/expected
    for i in arrClass:
        chiSquare += (v.loc[attr1][i]-(((v[i].sum())*(v1[attr1].sum()))/total))**2/(((v[i].sum())*(v1[attr1].sum()))/total)
        chiSquare += (v.loc[attr2][i]-(((v[i].sum())*(v1[attr2].sum()))/total))**2/(((v[i].sum())*(v1[attr2].sum()))/total)
    
    degreeOfFreedom = (len(v)-1)*(len(v1)-1)
    res += "Chi Square Value : "+ str(chiSquare) + " \n"
    res += "Degree of Freedom: " + str(degreeOfFreedom) + "\n"
    res +="\n"
    if chiSquare > degreeOfFreedom:
        res += "Attributes " + attr1 + ' and ' + attr2 + " are strongly correlated."
    else:
        res += "Attributes " + attr1 + ' and ' + attr2 + " are not correlated."
    return res

def pearsoncoef_with_x_y(col1,col2,classN):
    global df
    attr1 = col1
    attr2 = col2
    data = df

    sumx = 0
    for i in range(len(data)):
        sumx += data.loc[i, attr1]
    mean1 = sumx/len(data)
    
    sumx2 = 0
    for i in range(len(data)):
        sumx2 += (data.loc[i, attr1])*(data.loc[i, attr1])

    sumy2 = 0
    for i in range(len(data)):
        sumy2 += (data.loc[i, attr2])*(data.loc[i, attr2])

    sumy = 0
    for i in range(len(data)):
        sumy += data.loc[i, attr2]
    mean2 = sumy/len(data)
    
    sumxy=0
    for i in range(len(data)):
        sumxy += (data.loc[i, attr1])*(data.loc[i, attr2])

    sum = 0
    for i in range(len(data)):
        sum += (data.loc[i, attr1]-mean1)*(data.loc[i, attr2]-mean2)
    covariance = sum/len(data)

    n = len(data)
    pearsonCoeff = ((n*sumxy)-(sumx*sumy))/math.sqrt( ((n*sumx2)-(sumx*sumx))*((n*sumy2)-(sumy*sumy)))
    res = ""
    res += "Covariance value is "+str(covariance) + "\n "    
    res += "Correlation coefficient(Pearson coefficient) is "+str(pearsonCoeff) + "\n"
    if pearsonCoeff > 0:
        res += "Attributes " + attr1 + ' and ' + attr2 + " are positively correlated."
    elif pearsonCoeff < 0:
        res += "Attributes " + attr1 + ' and ' + attr2 + " are negatively correlated."
    elif pearsonCoeff == 0:
        res += "Attributes " + attr1 + ' and ' + attr2 + " are independant."
    return res

from sklearn.tree import _tree

def get_rules(tree, feature_names, class_names):
    tree_ = tree.tree_
    feature_name = [
        feature_names[i] if i != _tree.TREE_UNDEFINED else "undefined!"
        for i in tree_.feature
    ]

    paths = []
    path = []
    
    def recurse(node, path, paths):
        
        if tree_.feature[node] != _tree.TREE_UNDEFINED:
            name = feature_name[node]
            threshold = tree_.threshold[node]
            p1, p2 = list(path), list(path)
            p1 += [f"({name} <= {np.round(threshold, 3)})"]
            recurse(tree_.children_left[node], p1, paths)
            p2 += [f"({name} > {np.round(threshold, 3)})"]
            recurse(tree_.children_right[node], p2, paths)
        else:
            path += [(tree_.value[node], tree_.n_node_samples[node])]
            paths += [path]
            
    recurse(0, path, paths)

    # sort by samples count
    samples_count = [p[-1][1] for p in paths]
    ii = list(np.argsort(samples_count))
    paths = [paths[i] for i in reversed(ii)]
    
    rules = []
    for path in paths:
        rule = "if "
        
        for p in path[:-1]:
            if rule != "if ":
                rule += " and "
            rule += str(p)
        rule += " then "
        if class_names is None:
            rule += "response: "+str(np.round(path[-1][0][0][0],3))
        else:
            classes = path[-1][0][0]
            l = np.argmax(classes)
            rule += f"class: {class_names[l]} (proba: {np.round(100.0*classes[l]/np.sum(classes),2)}%)"
        rule += f" | based on {path[-1][1]:,} samples"
        rules += [rule]
        
    return rules 


def decisionTree(col1,op):
    global df
    data = df
    cols = []
    for i in df.columns:
        cols.append(i)
    cols.remove(col1)
    d = data
    #Unique Class Names of Given Attribute
    arrClass = data[col1].unique()

    f_names = []
    c_names = []
    f_names = cols
    print(f_names)
    for c in arrClass:
        c_names.append(str(c))
    print(type(c_names))
    
    df = data      
    dft = data.drop(col1, axis=1)
    print(dft)

    #Decision Tree Classifier Train : 80%  Test : 20%
    X_train, X_test, Y_train, Y_test = train_test_split(dft, df[col1], test_size=0.2, random_state=1)
    clf = DecisionTreeClassifier(max_depth = 3, random_state = 0,criterion="gini")
    model = clf.fit(X_train, Y_train)
    Y_predicted = clf.predict(X_test)

    if(op=="rules"):
        rules = get_rules(clf, f_names, c_names)
        res = "Rules Extracted : \n"
        for r in rules:
            print(r)
            res +="\n"+r
        c_matrix = confusion_matrix(Y_test,Y_predicted)
        res += "\nAccuracy of Extracted Rules : " + str(accuracy_score(Y_test,Y_predicted))
        #print ("Accuracy : ",accuracy_score(Y_test,Y_predicted)*100)
        print(c_matrix)

        res += "\nConfusion Matrix : \n"+ str(c_matrix)
        print(classification_report(Y_test,Y_predicted))
        res += "\n"+ str(classification_report(Y_test,Y_predicted))
        return res

    if(op=="gini"):
        
        clf = DecisionTreeClassifier(max_depth = 3, random_state = 0,criterion="gini")
        model = clf.fit(X_train, Y_train)
        Y_predicted = clf.predict(X_test)
        Y_test = Y_test.to_numpy()

        print(type(Y_predicted),type(Y_test))

        print(Y_predicted, "predicted", len(Y_predicted), Y_test, "Y_test", len(Y_test))
        c_matrix = confusion_matrix(Y_test,Y_predicted)
        res = ""
        res += "Accuracy by Gini Index : " + str(accuracy_score(Y_test,Y_predicted))
        #print ("Accuracy : ",accuracy_score(Y_test,Y_predicted)*100)
        print(c_matrix)

        res += "\nConfusion Matrix : "+ str(c_matrix)
        print(classification_report(Y_test,Y_predicted))
        res += "\n"+ str(classification_report(Y_test,Y_predicted))
        ax = plt.subplot()
        sns.heatmap(c_matrix, annot=True, fmt='g', ax=ax)

        #Confusion Matrix 
        ax.set_xlabel('Predicted labels')
        ax.set_ylabel('True labels') 
        ax.set_title('Confusion Matrix')
        ax.xaxis.set_ticklabels(c_names)
        ax.yaxis.set_ticklabels(c_names)
        
        #Decision Tree
        text_representation = tree.export_text(clf)
        # print(text_representation)
        print(f_names,c_names)
        print(type(f_names),type(c_names))
        
        #Visualization of Tree
        fig = plt.figure(figsize=(25,20))
        _ = tree.plot_tree(clf, feature_names=f_names, class_names=c_names,filled=True)
        plt.show()
        print("RES: ",res)
        return res
    
    if(op=="gain"):
        X_train, X_test, Y_train, Y_test = train_test_split(dft, df[col1], test_size=0.2, random_state=100)
        clf = DecisionTreeClassifier(max_depth = 3, random_state = 0,criterion="entropy")
        model = clf.fit(X_train, Y_train)
        Y_predicted = clf.predict(X_test)
        Y_test = Y_test.to_numpy()

        print(type(Y_predicted),type(Y_test))

        print(Y_predicted, "predicted", len(Y_predicted), Y_test, "Y_test", len(Y_test))
        c_matrix = confusion_matrix(Y_test,Y_predicted)
        res = ""
        res += "Accuracy by Gain Ratio : " + str(accuracy_score(Y_test,Y_predicted))
        #print ("Accuracy : ",accuracy_score(Y_test,Y_predicted)*100)
        print(c_matrix)

        res += "\nConfusion Matrix : "+ str(c_matrix)
        print(classification_report(Y_test,Y_predicted))
        res += "\n"+ str(classification_report(Y_test,Y_predicted))
        ax = plt.subplot()
        sns.heatmap(c_matrix, annot=True, fmt='g', ax=ax)

        #Confusion Matrix 
        ax.set_xlabel('Predicted labels')
        ax.set_ylabel('True labels') 
        ax.set_title('Confusion Matrix')
        ax.xaxis.set_ticklabels(c_names)
        ax.yaxis.set_ticklabels(c_names)
        
        #Decision Tree
        text_representation = tree.export_text(clf)
        # print(text_representation)
        print(f_names,c_names)
        print(type(f_names),type(c_names))
        
        #Visualization of Tree
        fig = plt.figure(figsize=(25,20))
        _ = tree.plot_tree(clf, feature_names=f_names, class_names=c_names,filled=True)
        plt.show()
        print("RES: ",res)
        return res



def normalization(col1,col2,col3,op):
    global df 
    data = df
    d = df
    window2 = Tk()
    window2.title("Normalization")
    window2.geometry("500x500")
    cols = []
    for i in data.columns:
        cols.append(i)
    attr1 = col1
    attr2 = col2 
    operation = op
    if operation == "mnorm":
        n = len(data)
        arr1 = []

        #Finding minimum and maximum values
        for i in range(len(data)):
            arr1.append(data.loc[i, attr1])
        arr1.sort()
        min1 = min(arr1)
        max1 = max(arr1)
        
        arr2 = []
        for i in range(len(data)):
            arr2.append(data.loc[i, attr2])
        arr2.sort()
        min2 = min(arr2)
        max2 = max(arr2)
        
        #Applying formula
        for i in range(len(data)):
            d.loc[i, attr1] = ((data.loc[i, attr1]-min1)/(max1-min1))
        
        for i in range(len(data)):
            d.loc[i, attr2] = ((data.loc[i, attr2]-min2)/(max2-min2))

    elif operation == "znorm":
        sum = 0
        #Calculation of mean
        for i in range(len(data)):
            sum += data.loc[i, attr1]
        mean1 = sum/len(data)
        sum = 0

        #Calculation of Variance
        for i in range(len(data)):
            sum += (data.loc[i, attr1]-mean1)*(data.loc[i, attr1]-mean1)
        var1 = sum/(len(data))

        #Standard Deviation
        sd1 = math.sqrt(var1)
        
        sum = 0
        for i in range(len(data)):
            sum += data.loc[i, attr2]
        mean2 = sum/len(data)
        sum = 0
        for i in range(len(data)):
            sum += (data.loc[i, attr2]-mean2)*(data.loc[i, attr2]-mean2)
        var2 = sum/(len(data))
        sd2 = math.sqrt(var2)
        
        #Normalization by Z Score (value-mean)/std_dev
        for i in range(len(data)):
            d.loc[i, attr1] = ((data.loc[i, attr1]-mean1)/sd1)
        
        for i in range(len(data)):
            d.loc[i, attr2] = ((data.loc[i, attr2]-mean2)/sd2)

    elif operation == "dnorm":        
        j1 = 0
        j2 = 0
        n = len(data)
        arr1 = []
        for i in range(len(data)):
            arr1.append(data.loc[i, attr1])
        arr1.sort()
        max1 = arr1[n-1]
        
        arr2 = []
        for i in range(len(data)):
            arr2.append(data.loc[i, attr2])
        arr2.sort()
        max2 = arr2[n-1]
        
        while max1 > 1:
            max1 /= 10
            j1 += 1
        while max2 > 1:
            max2 /= 10
            j2 += 1
        
        for i in range(len(data)):
            d.loc[i, attr1] = ((data.loc[i, attr1])/(pow(10,j1)))
        
        for i in range(len(data)):
            d.loc[i, attr2] = ((data.loc[i, attr2])/(pow(10,j2)))
    
    Label(window2,text="Normalized Attributes", justify='center',height=2,fg="green").grid(column=1,row=8,padx=5,pady=8)         
    tv1 = ttk.Treeview(window2,height=15)
    tv1.grid(column=1,row=9,padx=5,pady=8)
    tv1["column"] = [attr1,attr2]
    tv1["show"] = "headings"
    for column in tv1["columns"]:
        tv1.heading(column, text=column)
    i = 0
    while i < len(data):
        tv1.insert("", "end", iid=i, values=(d.loc[i, attr1],d.loc[i, attr2]))
        i += 1
    sns.set_style("whitegrid")
    sns.FacetGrid(d, hue=col3, height=4).map(plt.scatter, attr1, attr2).add_legend()
    plt.title("Scatter plot")
    plt.show(block=True)
    window2.mainloop()

from sklearn.naive_bayes import GaussianNB
from sklearn import metrics


# def myann(op):
#     data = pd.read_excel('iris.xlsx')
#     df_norm = data[data.columns[1:-1]].apply(lambda x: (x - x.min()) / (x.max() - x.min()))
#     unique = np.unique(data.iloc[:,-1])
    
#     target = data.iloc[:,-1].replace(unique,range(len(unique)))
#     df = pd.concat([df_norm, target], axis=1)
#     size = 30
#     train_test_per = (100-size)/100.0
#     df['train'] = np.random.rand(len(df)) < train_test_per
#     train = df[df.train == 1]
#     train = train.drop('train', axis=1).sample(frac=1)
#     test = df[df.train == 0]
#     test = test.drop('train', axis=1)
#     X = train.values[:,:4]
#     targets = [[1,0,0],[0,1,0],[0,0,1]]
#     y = np.array([targets[int(x)] for x in train.values[:,-1]])

#     num_inputs = len(X[0])
#     hidden_layer_neurons = 5
#     np.random.seed(4)
#     w1 = 2*np.random.random((num_inputs, hidden_layer_neurons)) - 1

#     num_outputs = len(y[0])
#     w2 = 2*np.random.random((hidden_layer_neurons, num_outputs)) - 1

#     # TRAINING
#     learning_rate = 0.2 
#     error = []
#     for epoch in range(1000):
#         l1 = 1/(1 + np.exp(-(np.dot(X, w1))))
#         l2 = 1/(1 + np.exp(-(np.dot(l1, w2))))
#         er = (abs(y - l2)).mean()
#         error.append(er)
        
#         # BACKPROPAGATION 
#         # find contribution of error on each weight on the second layer
#         l2_delta = (y - l2)*(l2 * (1-l2))
#         w2 += l1.T.dot(l2_delta) * learning_rate
        
#         l1_delta = l2_delta.dot(w2.T) * (l1 * (1-l1))
#         w1 += X.T.dot(l1_delta) * learning_rate
    
#     #TEST
#     X = test.values[:,:4]
#     y = np.array([targets[int(x)] for x in test.values[:,-1]])

#     l1 = 1/(1 + np.exp(-(np.dot(X, w1))))
#     l2 = 1/(1 + np.exp(-(np.dot(l1, w2))))

#     np.round(l2,3)


#     y_pred = np.argmax(l2, axis=1) 
#     res = y_pred == np.argmax(y, axis=1)
#     correct = np.sum(res)/len(res)

#     test_df = test
#     test_df[['variety']] = test[['variety']].replace(range(len(unique)), unique)

#     test_df['Prediction'] = y_pred
#     test_df['Prediction'] = test_df['Prediction'].replace(range(len(unique)), unique)

#     acc = correct
    
   
#     cfm = confusion_matrix(test_df[['variety']], test_df[['Prediction']])
    
#     res = ""
#     res +='Confusion Matrix: \n' + str(cfm)+'\n'
#     res +='Recognition rate: '+str(acc*100)
#     res +='\nMisclassification rate'+str((1-acc)*100)+"\n\n"
#     res += '\n'+ str(classification_report(test_df[['variety']], test_df[['Prediction']]))

#     plt.title('Error Graph')
#     plt.plot(error)
#     plt.xlabel('Epoch')
#     plt.ylabel('Error')
#     plt.show()
#     return res

@eel.expose
def kmncluster(clust,itr):
    print(clust,itr)
    return kmeanscluster(int(clust),int(itr))

@eel.expose
def kmdcluster(clust,itr):
    print(clust,itr)
    return kmcall(int(clust),int(itr))

@eel.expose
def hicluster(clust,itr):
    print(clust,itr)
    return hcluster(int(clust))

from association_rules import generate_rule

def read_data_in_dict(filename):
    f = open(filename)
    lines = f.readlines()
    transactions = []
    items = lines[0].split(',')
    for line in lines[1:]:
        transactions.append(list(map(int,line.split(','))))
    data ={
        'items':items,
        'transactions':transactions
    }
    return data
    
@eel.expose
def aproiri(sup,conf):
    data = pd.read_csv('itemsets.csv')
    return generate_rule(data,float(sup),float(conf))
    


@eel.expose
def annClass(input):
    testList = input.split(",")
    print(testList)
    testdata = []
    for x in testList:
        if(x != ''):
            testdata.append(int(x))
    return basicANN(testdata)    

@eel.expose
def naive_baysian_classifier(k,test):
    df = pd.read_excel('iris.xlsx')
    dataset = df
    X = dataset.iloc[:, :-1].values
    y = dataset.iloc[:, 4].values

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=1)
     
    gnb = GaussianNB()
    gnb.fit(X_train, y_train)
    
    y_pred = gnb.predict(X_test)
    testList = test.split(",")
    print(testList)
    testdata = []
    for x in testList:
        if(x != ''):
            testdata.append(float(x))
    y_pre = gnb.predict([testdata])
    res = "Predicted Class : "+str(y_pre) 
    res +="\n Gaussian Naive Bayes model accuracy : " + str(accuracy_score(y_test, y_pred))
    c_matrix = confusion_matrix(y_test, y_pred)
    res += "\nConfusion Matrix : "+ str(c_matrix)
    print(classification_report(y_test, y_pred))
    res += "\n"+ str(classification_report(y_test, y_pred))
    return res

@eel.expose
def knn_classifier(k,test):
    df = pd.read_excel('iris.xlsx')
    dataset = df
    X = dataset.iloc[:, :-1].values
    y = dataset.iloc[:, 4].values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 0)
    scaler = StandardScaler()
    scaler.fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)
    classifier = KNeighborsClassifier(n_neighbors = int(k))
    classifier.fit(X_train, y_train)
    # print(X_train)
    testList = test.split(",")
    print(testList)
    testdata = []
    for x in testList:
        if(x != ''):
            testdata.append(float(x))
    testdata = scaler.transform([testdata])
    y_pred = classifier.predict(X_test)
    y_pred1 = classifier.predict(testdata)
    print(testdata)
    print(y_pred)
    res = "Predicted Class : "+str(y_pred1) 
    res +="\n KNN model accuracy : " + str(accuracy_score(y_test, y_pred))
    c_matrix = confusion_matrix(y_test, y_pred)
    res += "\nConfusion Matrix : \n"+ str(c_matrix)
    print(classification_report(y_test, y_pred))
    res += "\n"+ str(classification_report(y_test, y_pred))
    return res



# # Exposing the random_python function to javascript
@eel.expose	
def random_python():
	print("Random function running")
	return randint(1,100)


@eel.expose	
def getCol(path):
    global df
    df = pd.read_excel(path)
    colList = list(df.columns)
    return colList

@eel.expose
def regression_classifier(k,test):
    df = pd.read_excel('iris.xlsx')
    dataset = df
    X = dataset.iloc[:, :-1].values
    y = dataset.iloc[:, 4].values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 0)
    scaler = StandardScaler()
    scaler.fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)
    # instantiate the model
    logreg = LogisticRegression(random_state=0)
    # fit the model
    logreg.fit(X_train, y_train)
    testList = test.split(",")
    print(testList)
    testdata = []
    for x in testList:
        if(x != ''):
            testdata.append(float(x))
    testdata = scaler.transform([testdata])
    y_pred_test = logreg.predict(testdata)
    y_p_test = logreg.predict(X_test)
    print(y_pred_test)
    res = ""
    res +=str(y_pred_test)
    print("score: ",str(accuracy_score(y_test, y_p_test)))
    res +="\n Regression Model accuracy score: " + str(accuracy_score(y_test, y_p_test))
    c_matrix = confusion_matrix(y_test, y_p_test)
    res += "\nConfusion Matrix : "+ str(c_matrix)
    print(classification_report(y_test, y_p_test))
    res += "\n"+ str(classification_report(y_test,y_p_test))
    return res

@eel.expose	
def showData():
    os.system('python A1.py')
    return "DONE"


@eel.expose	
def getRes(col,op):
    if(op=="mean"):
        return findMean(col)
    if(op=="mode"):
        return findMode(col)
    if(op=="median"):
        return findMedian(col)
    if(op=="std-dev"):
        return findstddeviation(col)
    if(op=="midrange"):
        return findMidrange(col)
    if(op=="variance"):
        return findvariance(col)
    if(op=="fiveno"):
        return findfive_number_summary(col)
    if(op=="quantile"):
        return calc_quantile(col)
    if(op=="qrange"):
        return calc_quantile_range(col)
    if(op=="range"):
        return findrange(col)
    if(op=="qplot"):
        qq_plot(col)
    if(op=="boxplot"):
        box_plot(col)
    if(op=="splot"):
        scatter_plot(col)
    if(op=="histo"):
        histogram(col)
    if(op=="gini" or op=="rules" or op=="gain"):
        # print(decisionTree(col,op))
        return decisionTree(col,op)
    if(op=="knn"):
        knn_classifier(col,op)
    


@eel.expose
def getRes2(col1,col2,col3,op):
    if(op=="chi"):
        return chi2test_with_x_y(col1,col2,col3)
    if(op=="pear"):
        return pearsoncoef_with_x_y(col1,col2,col3)
    if(op=="mnorm"):
        normalization(col1,col2,col3,op)
    if(op=="znorm"):
        normalization(col1,col2,col3,op)
    if(op=="dnorm"):
        normalization(col1,col2,col3,op)


    


# Start the index.html file
eel.start("kmean.html")
