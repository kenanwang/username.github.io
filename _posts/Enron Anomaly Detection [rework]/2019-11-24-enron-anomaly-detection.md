---
layout: post
title: Enron Anomaly Detection using Neural Network Analysis
categories: [End to End Projects]
tags: [Hopfield Memory Network, Network Analysis, NetworkX]
---
In this project I am replicating analysis from a research paper found below. This project maps a social network as a neural network, precisely a Hopfield memory network. We feed the Hopfield memory network features that represent anomalously high amounts of activity. We use a Hebbian learning rule to train our memory network. The Hebbian rule "remembers" nodes that fire together.  Thus, our memory network algorithm takes the input data of individuals who are displaying elevated activity, and extracts an even more precise signal of specific subnetworks of elevated activity occurring together.  

I use the Enron Corpus, found here: https://www.kaggle.com/wcukierski/enron-email-dataset, to try to replicate the analysis from this paper:

Volodymyr Miz, Benjamin Ricaud, Kirell Benzi, and Pierre Vandergheynst. 2019. Anomaly detection in the dynamics of web and social networks. In Proceedings of May 13-17 (TheWebConf 2019). , 10 pages. https://doi.org/10.475/123_4

Thanks to Volodymyr Miz who generously helped me troubleshoot my analysis and who has many helpful [online resources](http://blog.miz.space/research/2019/02/13/anomaly-detection-in-dynamic-graphs-and-time-series-networks/) to understand the research as well.

They were able to detect four periods of anamoly within the corpus data by treating the social network of email communications as a Hopfield Memory Network. They start by building a graph of email addresses as nodes and emails sent as edges. The attributes of the graph they built are described more specifically here:
https://zenodo.org/record/1342353#.Xap5medKjMJ

They use this paper to as the ground level truth that they compare their four periods of anamalous activity against:
Heng Wang, Minh Tang, Youngser Park, and Carey E Priebe. 2014. Locality statistics for anomaly detection in time series of graphs. IEEE Transactions on Signal Processing 62, 3 (2014), 703–717.

In order to better understand how to clean the data I also followed:

Shetty, Jitesh & Adibi, Jafar. (2004). The Enron email dataset database schema and brief statistical report. http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.296.9477&rep=rep1&type=pdf

## Table of Contents:

I. Attempt to duplicate the graph built by Miz, Ricaud, Benzi, Vandergheynst (2019), now called official data.

II. Attempt to duplicate the spatio-temporal analysis that they perform to catch the anomalies.

III. Analysis of the activity of the anomalous clusters

## 0. Import Libraries


```python
# Libraries for data analysis
import numpy as np
import pandas as pd

from matplotlib import pyplot as plt
%matplotlib inline

import seaborn as sns
sns.set_style('darkgrid')

import re
from datetime import datetime
from collections import defaultdict

import networkx as nx
from nxviz import CircosPlot

import random
from operator import sub
import math
import statistics
import ast
```

# I. Duplicate Graph


```python
# Defining a function to take the message column from the Dataframe of emails and extract three Pandas Series containing: the date of the email, the addresses that the email is from, the addresses that the email is sent to
# The data, from, and to data of the emails is what we need to generate the graph of our social network
def get_date_from_to(Series):
    # Generate three empty Pandas Series to hold the data we will output
    result_date = pd.Series(index=Series.index)
    result_from = pd.Series(index=Series.index)
    result_to = pd.Series(index=Series.index)
    # Since this process takes some time with the full data, this is a tracking function
    for row, message in enumerate(Series):
        if row % 10000 == 0:
            print(f'Row: {row} starting at {datetime.now()}.')
        # The message data is fairly simple and split predictable by lines
        message_words = message.split('\n')
        # Data is generally in the second line
        if 'Date:' in message_words[1]:
            result_date[row] = message_words[1].replace('Date:', '')
        else:
            result_date[row] = np.nan
        # From addresses are generally in the third line, below is a regex to find email addresses
        if 'From:' in message_words[2]:
            result_from[row] = re.findall('[\w\.-]+@[\w\.-]+\.\w+', message_words[2])
        else:
            result_from[row] = np.nan
        # To addresses are generally in the fourth line, below is a regex to find email addresses
        if 'To:' in message_words[3]:
            result_to[row] = re.findall('[\w\.-]+@[\w\.-]+\.\w+', message_words[3])
        else:
            result_to[row] = np.nan
    # We convert the data times to the format of days after Jan 1, 1999 consistent with how data is stored by Miz, Ricaud, Benzi, Vandergheynst (2019).
    print('Converting Dates.')
    result_date = pd.to_datetime(result_date)
    result_date = result_date - datetime(1999,1,1) # As in the paper, will drop values before Jan 1, 1999
    return result_date, result_from, result_to
```


```python
# This generates a graph with nodes for email addresses, edges if an email is sent once, and a time lapse dictionary associated with each node capturing how many emails were sent on what days
def gen_graph(date_from_to):
    G = nx.Graph()
    for index, row in date_from_to.iterrows():
        for sender in row.senders:
            if sender not in G:
                G.add_node(sender)
            for recipient in row.recipients:
                if recipient not in G:
                    G.add_node(recipient)
                if not(G.has_edge(sender, recipient)):
                    G.add_edge(sender, recipient)
                G.node[sender][row.date.days] = G.node[sender].get(row.date.days, 0) + 1
    return G
```


```python
# This maps the node labels based on email addresses to integer labels
def map_nodes_to_int(G):
    nodes = G.nodes
    mapping = {}
    node_index = 0
    for node in nodes:
        mapping[node] = node_index
        node_index+=1
    H = nx.relabel_nodes(G, mapping)
    return H, mapping  
```


```python
# Miz, Ricaud, Benzi, Vandergheynst (2019) filtered out any email addresses that didn't send more than 3 emails
def filter_graph_down(G, min_emails=4):
    nodes_to_keep = []
    nodes = G.nodes
    for key, values in nodes.items():
        if sum(values.values()) >= min_emails:
            nodes_to_keep.append(key)
    sub_G = G.subgraph(nodes_to_keep)
    return sub_G
```


```python
# Reading in emails
emails = pd.read_csv('emails.csv')
```


```python
# Dropping dups and auto created emails
# Generating table of dates, senders, receivers
emails_nodups = emails[~emails.file.str.contains('discussion_thread')]
emails_noautos = emails_nodups[~emails_nodups.file.str.contains('all_documents')]
emails_noautos = emails_noautos.reset_index()
date_from_to = pd.DataFrame()
date_from_to['date'], date_from_to['senders'], date_from_to['recipients'] = get_date_from_to(emails_noautos.message)
print(len(date_from_to))
```

    Row: 0 starting at 2019-11-19 14:53:53.829113.
    Row: 10000 starting at 2019-11-19 14:53:54.887632.
    Row: 20000 starting at 2019-11-19 14:53:55.942886.
    Row: 30000 starting at 2019-11-19 14:53:56.442161.
    Row: 40000 starting at 2019-11-19 14:53:57.133102.
    Row: 50000 starting at 2019-11-19 14:53:58.193815.
    Row: 60000 starting at 2019-11-19 14:53:58.761066.
    Row: 70000 starting at 2019-11-19 14:53:59.760541.
    Row: 80000 starting at 2019-11-19 14:54:00.186859.
    Row: 90000 starting at 2019-11-19 14:54:00.736105.
    Row: 100000 starting at 2019-11-19 14:54:01.146841.
    Row: 110000 starting at 2019-11-19 14:54:01.649933.
    Row: 120000 starting at 2019-11-19 14:54:02.440419.
    Row: 130000 starting at 2019-11-19 14:54:03.123122.
    Row: 140000 starting at 2019-11-19 14:54:03.575431.
    Row: 150000 starting at 2019-11-19 14:54:04.095484.
    Row: 160000 starting at 2019-11-19 14:54:04.594701.
    Row: 170000 starting at 2019-11-19 14:54:05.261858.
    Row: 180000 starting at 2019-11-19 14:54:05.858063.
    Row: 190000 starting at 2019-11-19 14:54:06.220688.
    Row: 200000 starting at 2019-11-19 14:54:06.904033.
    Row: 210000 starting at 2019-11-19 14:54:07.341307.
    Row: 220000 starting at 2019-11-19 14:54:07.958553.
    Row: 230000 starting at 2019-11-19 14:54:08.361247.
    Row: 240000 starting at 2019-11-19 14:54:08.764725.
    Row: 250000 starting at 2019-11-19 14:54:09.203054.
    Row: 260000 starting at 2019-11-19 14:54:09.606893.
    Row: 270000 starting at 2019-11-19 14:54:10.039794.
    Row: 280000 starting at 2019-11-19 14:54:10.791288.
    Row: 290000 starting at 2019-11-19 14:54:11.286718.
    Row: 300000 starting at 2019-11-19 14:54:12.337010.
    Row: 310000 starting at 2019-11-19 14:54:12.938221.
    Row: 320000 starting at 2019-11-19 14:54:13.349353.
    Row: 330000 starting at 2019-11-19 14:54:13.748598.
    Converting Dates.
    330689



```python
# Filtering out naan entries
# Filtering out entries before Jan 1, 1999
# Filtering out entries after July 31, 2002
print(len(date_from_to))
date_from_to.dropna(inplace=True)
date_from_to = date_from_to[date_from_to.date >= pd.Timedelta(0)]
date_from_to = date_from_to[date_from_to.date <= pd.Timedelta(days=1448)] # Dropping emails after 31 July 2002
print(len(date_from_to))
```

    330689
    313191



```python
# Generating initial Graph
G_init = gen_graph(date_from_to)
```


```python
# Filtering out nodes with 3 or fewer emails sent during time period
G_emails = filter_graph_down(G_init, min_emails=4)
```

    34897
    87780



```python
G_mapped, mapping = map_nodes_to_int(G_emails)
```


```python
print(nx.number_of_nodes(G_mapped))
print(nx.number_of_edges(G_mapped))
```

    7125
    47097


The graph generated by Miz, Ricaud, Benzi, Vandergheynst (2019), had 6600 nodes and 50,897 edges. I wasn't able to perfectly duplicate this result, but in order to get close I had to do things like drop automatically sent emails and I had to drop some duplicated emails as in Shetty, Adibi (2004). The day numbers differ slightly between our two sets as well (by 3 days), I'm not sure why this is. In the next part I will use both my data and the data provided by Miz, Ricaud, Benzi, Vandergheynst (2019) to perform the ML analysis.

Note: Turns out that the reason my data is off is because I kept all emails that either sent or received 3 emails. Miz, Ricaud, Benzi, Vandergheynst (2019) only kept nodes that sent 3 emails. I'm going to use their data from now moving forward. But this was a good exercise. Below I import their data.


```python
nodes_df = pd.read_csv('nodes.csv', header=None, names=['node_dict'], index_col=0)
edges_df = pd.read_csv('edges.csv')
print(nodes_df.head())
print(edges_df.head())
```

                                               node_dict
    0                                           {687: 6}
    1  {1024: 1, 1089: 1, 1132: 1, 1100: 1, 1165: 1, ...
    2                                   {611: 4, 612: 2}
    3                         {1004: 2, 1005: 4, 967: 2}
    4  {897: 3, 900: 1, 903: 1, 905: 1, 906: 1, 962: ...
       From    To  Count
    0     0  1760      3
    1     2  2692      6
    2     3  2719      2
    3     3  3118      2
    4     3  5211      2



```python
nodes_list = [(index, ast.literal_eval(row.node_dict)) for index, row in nodes_df.iterrows()]
edges_list = [(row.From, row.To) for index, row in edges_df.iterrows()]
print(nodes_list[:5])
print(edges_list[:5])
```

    [(0, {687: 6}), (1, {1024: 1, 1089: 1, 1132: 1, 1100: 1, 1165: 1, 1075: 1, 1109: 1, 1081: 1, 1114: 1}), (2, {611: 4, 612: 2}), (3, {1004: 2, 1005: 4, 967: 2}), (4, {897: 3, 900: 1, 903: 1, 905: 1, 906: 1, 962: 1, 910: 1, 920: 6, 792: 3, 794: 1, 795: 2, 925: 3, 802: 1, 931: 3, 805: 1, 806: 1, 807: 1, 938: 1, 939: 1, 940: 1, 941: 1, 942: 1, 945: 1, 946: 1, 947: 3, 948: 4, 990: 1, 953: 1, 826: 1, 955: 6, 956: 11, 960: 1, 833: 2, 961: 5, 966: 1, 840: 1, 973: 5, 974: 6, 975: 6, 850: 3, 855: 1, 856: 3, 996: 1, 991: 3, 862: 1, 863: 3, 864: 1, 868: 1, 869: 3, 998: 3, 872: 1, 935: 1, 876: 11, 877: 2, 878: 3, 879: 1, 882: 4, 884: 5, 886: 4, 890: 3})]
    [(0, 1760), (2, 2692), (3, 2719), (3, 3118), (3, 5211)]



```python
# Creating graph with official data
# Note: I get 50 897 directed edges (number in their paper), but 40777 undirected edges
# since the following analysis uses undirected edges I go ahead with that
G_mapped = nx.Graph()
G_mapped.add_nodes_from(nodes_list)
G_mapped.add_edges_from(edges_list)
print(nx.number_of_nodes(G_mapped))
print(nx.number_of_edges(G_mapped))
```

    6600
    40777


# II. Implementing Algorithms from Paper

Miz, Ricaud, Benzi, Vandergheynst (2019) treats the social network of people sending emails at Enron as a graph where the nodes represent email addresses (that sent more than 3 emails during the time in question) and edges exist between email addresses if an email was ever sent between the email addresses.

This graph (which represents a social network) is then treated like a memory network. Specifically the network is a Hopfield memory network using the Hebbian learning rule. A Hopfield network stores memories (and can be recalled in using partial informatino). The Hebbian rule creates memories when nodes (neurons) fire together, in this case an abnormal level of emailing is what is registered as a neuron firing.

Miz, Ricaud, Benzi, Vandergheynst (2019) go through 4 stages of when implementing the algorithm just described:

- They extract features by finding abnormal bursts of emails from one person. These basically correspond in future steps to a "neuron firing" within the memory field.

- They train Hopfield memory fields from those bursts using the Hebbian rule, which means connected people sending abnormally higher levels of emails together gets trained into the field as a "memorable event". A new Hopfield memory field is trained each month in this case, so they're attempting to learn new "memories" each month.

- They explore the Hopfield memory fields from certain months and look for the largest connected subgraph from that memory field and treat that as a "memorable" co-occurence of abnormally high email behavior

- They look at that subgraph of people and analyze their email volume to confirm that there is in fact an anomaly of emails during that time, this is compared to four events that happened in real life where higher email traffic would be expected

## II.a. Extract features and filter nodes


```python
# This generates an array displaying the time series activity of each node in the rows of the array
# This matrix of time series activity will be used to generate the mean and std of activity for each time step
def gen_signal_array(G, t_min=0, t_max=1448):
    nodes = G.nodes(data=True)
    signal_array = np.zeros(shape=(len(nodes),t_max+1-t_min))
    for t in range(t_min, t_max+1):
        for node, data in nodes:
            signal_array[node][t-t_min] = data.get(t,0)
    return signal_array
```


```python
# From the matrix of time series activity this function generates the mean and std of each time step
def gen_means_and_stds(signal_array):
    means = np.apply_along_axis(np.mean, 0, signal_array)
    stds = np.apply_along_axis(np.std, 0, signal_array)
    return means, stds
```


```python
# Keep only nodes that have bursts above a certain minimum number of bursts
# Only keep activity that is anamolous (a burst)
# A burst is defined as activity in a time step a certain number of STDs above the mean for that timestep
def filter_potential_anomalies(G, t_min, t_max, burst_threshold, min_bursts):
    signal_array = gen_signal_array(G, t_min, t_max)
    means, stds = gen_means_and_stds(signal_array)
    nodes = G.nodes
    nodes_to_keep = []
    activity = {}
    for node, node_dict in nodes.items():
        bursts = 0
        activity_to_keep = {}
        #freq = 0
        total = 0
        for t in range(t_min, t_max+1):
            if node_dict.get(t,0)>0:
                total+=node_dict[t]
                #freq+=1
                #activity_to_keep[t] = node_dict[t]
        for t in range(t_min, t_max+1):
            if node_dict.get(t,0) > means[t-t_min] + stds[t-t_min]*burst_threshold:
                bursts+=1
                activity_to_keep[t] = node_dict[t]
        if bursts >= min_bursts:
            nodes_to_keep.append(node)
            activity[node] = dict([(k, v/total) for k,v in activity_to_keep.items()])
    #H=G.subgraph(nodes_to_keep)
    H = nx.Graph()
    H.add_nodes_from(G.subgraph(nodes_to_keep).nodes)
    H.add_edges_from(G.subgraph(nodes_to_keep).edges)
    nx.set_node_attributes(H, activity)
    return H
```

Using the above function I extract features. In this case the features are the nodes with enough potentially anomalous activity as well as the potentially anomalous activity itself. By filtering out only these nodes and this activity I am extracting the useful features that will be used for training later with the Hopfield memory network.


```python
# December graph
G_dec = filter_potential_anomalies(G_mapped, t_min=330, t_max=361, burst_threshold=3, min_bursts=2)
```


```python
print(nx.number_of_nodes(G_dec))
print(nx.number_of_edges(G_dec))
```

    73
    276



```python
# April graph
G_apr = filter_potential_anomalies(G_mapped, t_min=818, t_max=848, burst_threshold=5, min_bursts=3)
```


```python
print(nx.number_of_nodes(G_apr))
print(nx.number_of_edges(G_apr))
```

    46
    213



```python
# May graph
G_may = filter_potential_anomalies(G_mapped, t_min=848, t_max=886, burst_threshold=5, min_bursts=3)
```


```python
print(nx.number_of_nodes(G_may))
print(nx.number_of_edges(G_may))
```

    89
    557



```python
# August graph
G_aug = filter_potential_anomalies(G_mapped, t_min=940, t_max=971, burst_threshold=6, min_bursts=3)
```


```python
print(nx.number_of_nodes(G_aug))
print(nx.number_of_edges(G_aug))
```

    55
    190


## II.b. Train Hopfield Network and extracting Largest Connected Component Subgraph


```python
# We use activity in this network to train metaphoric memories into a function called a Hopfield Memory Network
# It is called a memory network because if you have partial data and you iteratively apply the Hopfield Memory Network to that partial data you would "recall" information from the training data
# By looking at the active nodes in the Hopfield Memory Network you can get an idea of what Nodes are displaying noteworthy behavior within the training data
def train_hopfield(G, min_time, max_time, sparsity_parameter=0.5, forgetting_parameter=0):
    H = G.copy()
    edges = H.edges
    edge_weights = {}
    sim = {}
    for i, j in edges:
        node_dict_i = G.nodes(data=True)[i]
        node_dict_j = G.nodes(data=True)[j]
        edge_weights[(i,j)] = 0
        # Train by stepping through time and use Hebbian learning rule (using the similarity function defined)
        for t in range(min_time, max_time+1):
            activities = (node_dict_i.get(t,0), node_dict_j.get(t,0))
            if activities == (0,0):
                sim[(i,j,t)] = 0
            else:
                sim[(i,j,t)] = min(activities)/max(activities)
            if sim[(i,j,t)] > sparsity_parameter:
                edge_weights[(i,j)] += sim[(i,j,t)]
            else:
                edge_weights[(i,j)] -forgetting_parameter*sim[(i,j,t)]
    nx.set_edge_attributes(H, name='weight', values=edge_weights)
    return H
```


```python
# Filter out edges with low weight in the Hopfield Memory Network
def filter_edges_by_weight(G, weight_limit=0):
    H = G.copy()
    to_remove = []
    edges = H.edges(data=True)
    for i, j, data in edges:
        if data['weight'] <= weight_limit:
            to_remove.append((i,j))
    H.remove_edges_from(to_remove)
    return H
```

Below we train the Hopfield memory networks for each month, we filter out the low weight edges, after that we look for the largest connected subgraph in the trained network. Hypothetically this would represent a group of people all connected together who are simultaneously (we know it's simultaneous because we're using a Hebbian learning rule) displaying anomalous activity. Such a cluster would likely display the most interesting activity of all people within that month.


```python
# Training the Hopfield network for December 1999 (first anomaly)
G_trained_dec = train_hopfield(G_dec, min_time=330, max_time=361)
```


```python
# Finding the largest connected component subgraph of the trained Graph for December
G_trained_dec_filtered = filter_edges_by_weight(G_trained_dec)
Gc_dec = max(nx.connected_component_subgraphs(G_trained_dec_filtered), key=len)
print(nx.number_of_nodes(Gc_dec))
```

    45



```python
# Training the Hopfield network for April 2001 (second anomaly)
G_trained_apr = train_hopfield(G_apr, min_time=818, max_time=848)
```


```python
# Finding the largest connected component subgraph of the trained Graph for December
G_trained_apr_filtered = filter_edges_by_weight(G_trained_apr, weight_limit=1)
Gc_apr = max(nx.connected_component_subgraphs(G_trained_apr_filtered), key=len)
print(nx.number_of_nodes(Gc_apr))
```

    38



```python
# Training the Hopfield network for May 2001 (third and biggest anomaly)
G_trained_may = train_hopfield(G_may, min_time=848, max_time=886)
```


```python
# Finding the largest connected component subgraph of the trained Graph for May 2001
G_trained_may_filtered = filter_edges_by_weight(G_trained_may, weight_limit=1)
Gc_may = max(nx.connected_component_subgraphs(G_trained_may_filtered), key=len)
print(nx.number_of_nodes(Gc_may))
```

    54



```python
# Training the Hopfield network for Aug 2001 (fourth anomaly)
G_trained_aug = train_hopfield(G_aug, min_time=940, max_time=971)
```


```python
# Finding the largest connected component subgraph of the trained Graph for May 2001
G_trained_aug_filtered = filter_edges_by_weight(G_trained_aug)
Gc_aug = max(nx.connected_component_subgraphs(G_trained_aug_filtered), key=len)
print(nx.number_of_nodes(Gc_aug))
```

    41


## III. Analysis of Anomalous Clusters

Now that we've found the anamolous cluster of people for each month let's confirm that their activity is quantitatively interesting. One way of doing this is comparing the cumulative activity of the people within an anamolous cluster for their month of interest vs their activity at all other times.


```python
def gen_activity_graph(G_original, cluster, t_min=0, t_max=1448):
    original_times = []
    for node, node_dict in cluster.nodes(data=True):
        for t,v in node_dict.items():
            original_times.append(t)
    cluster = G_original.subgraph(cluster)
    time = range(t_min, t_max+1)
    array = np.zeros(shape=(len(time)))
    for t in time:
        for _, node_dict in cluster.nodes(data=True):
            array[t-t_min]+=node_dict.get(t,0)
    plt.plot(array)
    plt.axvspan(min(original_times),max(original_times),color='red', alpha=0.3)
    return array
```


```python
dec_array = gen_activity_graph(G_mapped, Gc_dec)
```


![png](output_50_0.png)


The anamolous cluster of people for December 1999 indeed sent notably more emails in December 1999 (highlighted in red) than at any other time. At this time Enron was attempting to push through a deal with the help of Merrill-Lynch before the end of their financial quarter. This activity has since been [investigated by regulators](https://www.nytimes.com/2002/10/04/business/1999-transaction-with-enron-keeps-dogging-merrill-lynch.html).


```python
apr_array = gen_activity_graph(G_mapped, Gc_apr)
```


![png](output_52_0.png)


The anamolous cluster of people for April 2001 indeed sent notably more emails in April 2001 (highlighted in red) than at any other time. This corresponds with the time period when Enron CEO Jeffrey Skilling was [publically questioned](https://en.wikipedia.org/wiki/Enron_scandal#Timeline_of_downfall) about their earnings.


```python
may_array = gen_activity_graph(G_mapped, Gc_may)
```


![png](output_54_0.png)


The anamolous cluster of people for May 2001 indeed sent notably more emails in May 2001 (highlighted in red) than at any other time. This corresponds with "formal notice of closure and termination of Enron’s single largest foreign investment, the Dabhol
Power Company in India." (Heng Wang, Minh Tang, Youngser Park, and Carey E Priebe. 2014.)


```python
aug_array = gen_activity_graph(G_mapped, Gc_aug)
```


![png](output_56_0.png)


My August cluster does not seem to display anamolous activity in the time series quantitative data as clearly as the data produced by Miz, Ricaud, Benzi, Vandergheynst (2019) I seem to be producing a similar signal for august but I have a lot of additional noise in other time periods that they do not. I'm not quite able to diagnose why this is the case but my algorithm is perfectly capable of finiding the signal from the noise in the other months. What follows is an analysis of the difference in the nodes between my data and the official data for the month of August.  

August 2001 is when Enron CEO, Jeffrey Skilling, resigned.


```python
id_email = pd.read_csv('id-email.csv', header=None, names=['id','email'], index_col='id')
id_email.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>email</th>
    </tr>
    <tr>
      <th>id</th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1800flowers@shop2u.com</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2000greetings@postalmanager.com</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2740741@skytel.com</td>
    </tr>
    <tr>
      <th>3</th>
      <td>40ees@enron.com</td>
    </tr>
    <tr>
      <th>4</th>
      <td>40enron@enron.com</td>
    </tr>
  </tbody>
</table>
</div>




```python
aug_emails = [email for email in map(lambda x: id_email.loc[x]['email'], Gc_aug.nodes)]
print(f'August emails: {aug_emails}')
```

    August emails: ['sgovenar@govadv.com', '40enron@enron.com', 'debra.perlingiere@enron.com', 'john.zufferli@enron.com', 'lorna.brennan@enron.com', 'sylvia.hu@enron.com', 'virginia.thompson@enron.com', 'chris.stokley@enron.com', 'marie.heard@enron.com', 'michelle.cash@enron.com', 'jeff.dasovich@enron.com', 'gerald.nemec@enron.com', 'jonathan.mckay@enron.com', 'monika.causholli@enron.com', 'michelle.lokay@enron.com', 'tim.belden@enron.com', 'kimberly.watson@enron.com', 'stacey.bolton@enron.com', 'alan.comnes@enron.com', 'diana.scholtes@enron.com', 'elizabeth.sager@enron.com', 'lynn.blair@enron.com', 'john.arnold@enron.com', 'glen.hass@enron.com', 'ken.powers@enron.com', 'dutch.quigley@enron.com', 'lisa.jacobson@enron.com', 'j.kaminski@enron.com', 'grace.rodriguez@enron.com', 'susan.mara@enron.com', 'kate.symes@enron.com', 'bill.williams@enron.com', 'darrell.schoolcraft@enron.com', 'paul.kaufman@enron.com', 'chris.dorland@enron.com']



```python
# Set of August emails for official data subtracting the set I got for August
set(['40enron@enron.com', 'john.sturn@enron.com', 'gretchen.hardeway@enron.com', 'greg.mann@enron.com', 'j.kaminski@enron.com', 'dave.perrino@enron.com', 'kimberly.watson@enron.com', 'larry.pavlou@enron.com', 'ron.matthews@enron.com', 'greg.frers@enron.com', 'ken.powers@enron.com', 'magdelena.cruz@enron.com', 'john.zufferli@enron.com', 'paul.donnelly@bakeratlas.com', 'ketchekl@bp.com', 'diana.scholtes@enron.com', 'jean.mrha@enron.com', 'eric.faucheaux@enron.com', 'bill.williams@enron.com', 'payables.ibuyit@enron.com', 'jeanie.slone@enron.com', 'monika.causholli@enron.com', 'tuckiejeff@hotmail.com', 'chris.stokley@enron.com', 'holden.salisbury@enron.com', 'jonathan.mckay@enron.com', 'dee.espinoza@travelpark.com', 'margaret.daffin@enron.com']) - set(aug_emails)
```




    {'dave.perrino@enron.com',
     'dee.espinoza@travelpark.com',
     'eric.faucheaux@enron.com',
     'greg.frers@enron.com',
     'greg.mann@enron.com',
     'gretchen.hardeway@enron.com',
     'holden.salisbury@enron.com',
     'jean.mrha@enron.com',
     'jeanie.slone@enron.com',
     'john.sturn@enron.com',
     'ketchekl@bp.com',
     'larry.pavlou@enron.com',
     'magdelena.cruz@enron.com',
     'margaret.daffin@enron.com',
     'paul.donnelly@bakeratlas.com',
     'payables.ibuyit@enron.com',
     'ron.matthews@enron.com',
     'tuckiejeff@hotmail.com'}




```python
apr_emails = [email for email in map(lambda x: id_email.loc[x]['email'], Gc_apr.nodes)]
print(f'April emails: {apr_emails}')
```

    April emails: ['chris.germany@enron.com', 'sgovenar@govadv.com', 'debra.perlingiere@enron.com', 'becky.spencer@enron.com', 'mary.cook@enron.com', 'janel.guerrero@enron.com', 'rhonda.denton@enron.com', 'bill.iii@enron.com', 'matthew.lenhart@enron.com', 'richard.sanders@enron.com', 'richard.shapiro@enron.com', 'tori.kuykendall@enron.com', 'outlook.team@enron.com', 'jeff.dasovich@enron.com', 'alan.comnes@enron.com', 'drew.fossum@enron.com', 'patrice.mims@enron.com', 'karen.denne@enron.com', 'stanley.horton@enron.com', 'kate.symes@enron.com', 'veronica.espinoza@enron.com', 'michael.tribolet@enron.com', 'vince.kaminski@enron.com', 'kay.mann@enron.com', 'steven.kean@enron.com', 'pete.davis@enron.com', 'dan.hyvl@enron.com', 'elizabeth.sager@enron.com', 'cara.semperger@enron.com', 'phillip.allen@enron.com', 'phillip.love@enron.com', 'mike.mcconnell@enron.com', 'sally.beck@enron.com', 'carol.clair@enron.com', 'susan.mara@enron.com', 'susan.scott@enron.com', 'enron.announcements@enron.com', 'darron.giron@enron.com', 'sara.shackleton@enron.com', 'kim.ward@enron.com', 'ccampbell@kslaw.com', 'tana.jones@enron.com', 'john.arnold@enron.com', 'evelyn.metoyer@enron.com', 'ray.alvarez@enron.com']



```python
set(apr_emails)&set(aug_emails)#&set(['40enron@enron.com', 'john.sturn@enron.com', 'gretchen.hardeway@enron.com', 'greg.mann@enron.com', 'j.kaminski@enron.com', 'dave.perrino@enron.com', 'kimberly.watson@enron.com', 'larry.pavlou@enron.com', 'ron.matthews@enron.com', 'greg.frers@enron.com', 'ken.powers@enron.com', 'magdelena.cruz@enron.com', 'john.zufferli@enron.com', 'paul.donnelly@bakeratlas.com', 'ketchekl@bp.com', 'diana.scholtes@enron.com', 'jean.mrha@enron.com', 'eric.faucheaux@enron.com', 'bill.williams@enron.com', 'payables.ibuyit@enron.com', 'jeanie.slone@enron.com', 'monika.causholli@enron.com', 'tuckiejeff@hotmail.com', 'chris.stokley@enron.com', 'holden.salisbury@enron.com', 'jonathan.mckay@enron.com', 'dee.espinoza@travelpark.com', 'margaret.daffin@enron.com'])
```




    {'alan.comnes@enron.com',
     'debra.perlingiere@enron.com',
     'elizabeth.sager@enron.com',
     'jeff.dasovich@enron.com',
     'john.arnold@enron.com',
     'kate.symes@enron.com',
     'sgovenar@govadv.com',
     'susan.mara@enron.com'}



I have overlap between my April and August email clusters. This same phenomenon does not occur in the official data. This would explain some of the noise in my august data, although it's still not clear what about my algorithm is causing this.
