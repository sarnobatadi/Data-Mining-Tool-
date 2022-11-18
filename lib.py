import eel
import pandas as pd
from collections import Counter
from random import randint
import math

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
        res+=str(j+1)+"th Quantile: "+str(ans[j]) + "\n"
        print(res)
    return res

def calc_quantile_range(col):
    global df 
    y = df[col]
    s_lst = sorted(y)
    ans=[]
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
    ans="Interquantile Range: "+str(ans[2]-ans[0])
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