import requests
from urllib.parse import urlparse, urljoin
from bs4 import BeautifulSoup
import pandas as pd
import numpy as np
from Graph import Graph
import eel

def init_graph(file):
        
	def split(line):
		str = ""
		flag = False
		for j in line:
			if not flag and j=='	':
				str = str+','
				flag = True
			else:
				str = str+j
		return str.split(',')

	f = open(file)
	lines = f.readlines()

	graph = Graph()

	for line in lines: 
		[parent, child] = split(line)
					
		graph.add_edge(parent, child)

		graph.sort_nodes()

	return graph


def PageRank_one_iter(graph, d):
	node_list = graph.nodes
	# print(node_list)
	for node in node_list:
		node.update_pagerank(d, len(graph.nodes))
	graph.normalize_pagerank()
	# print(graph.get_pagerank_list())
	# print()

def HITS_one_iter(graph):
	node_list = graph.nodes

	for node in node_list:
		node.update_auth()

	for node in node_list:
		node.update_hub()

	graph.normalize_auth_hub()


def HITS(graph, iteration=100):
	for i in range(iteration):
		HITS_one_iter(graph)
		# graph.display_hub_auth()
		# print()




def PageRank(iteration,graph, d):
	for i in range(int(iteration)):
		# print(i)
		PageRank_one_iter(graph, d)


def pgrank_res(itr,damp_fact):

	iteration = itr
	damping_factor = damp_fact
	file = "pgrank.txt"
	graph = init_graph(file)

	nodes = graph.nodes

	PageRank(iteration, graph, damping_factor)

	ranks_by_nodes = []
	page_ranks = graph.get_pagerank_list()

	for i in range(len(nodes)):
		ranks_by_nodes.append([nodes[i].name,[child.name for child in nodes[i].children],[parent.name for parent in nodes[i].parents],page_ranks[i]])


	df = pd.DataFrame(ranks_by_nodes,columns=["Node","Children","parents","Page Rank"])
	df = df.sort_values(by=["Page Rank","Node"],ascending=False)
	
	res = ""
	res += str(df.head(10))
	# table = st.table(df)

	res +="\nTotal page rank sum: "+str(np.sum(graph.get_pagerank_list()))
	print(res)
	return res


def HIT_res(itr):

	iteration = itr
	file = "pgrank.txt"
	graph = init_graph(file)

	HITS(graph,iteration)
	auth_list, hub_list = graph.get_auth_hub_list()

	nodes = [node.name for node in graph.nodes]

	my_data = []

	# print(hub_list)
	for i in range(len(nodes)):
		my_data.append([nodes[i],auth_list[i],hub_list[i]])

	df = pd.DataFrame(my_data,columns=["Node","Auth Value","Hub Value"])

	df = df.sort_values(["Auth Value","Hub Value"],ascending=False)
	# table = st.table(df)
	# for i in df:
	# 	print(i)
	# print(df)
	res = ""
	res += str(df.head(10))
	res += "Sum of Auth : "+ str(sum(auth_list)) + "\nSum of Hub List : " + str(sum(hub_list))
	print(res)
	# print(sum(auth_list)," ",sum(hub_list))
	return res

# HIT_res(1)
pgrank_res(1,0.15)