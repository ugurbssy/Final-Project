#!/usr/bin/env python
# coding: utf-8

# Market Basket Analysis

# By analysing the buying behaviour of customers, try to find out which are the products that are bought together by the customers.

# In[1]:


pip install apyori


# In[7]:


import pandas as pd
import numpy as np 
import seaborn as sns
import matplotlib.pyplot as plt
from apyori import apriori


# In[8]:


import sys,getopt
import requests
import csv
import Orange
from Orange.data import Table,Domain, DiscreteVariable, ContinuousVariable
from orangecontrib.associate.fpgrowth import *
from scipy import sparse
import scipy.stats as ss

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec 
import seaborn as sns
from PIL import Image


# In[9]:


df = pd.read_csv(r"C:\Users\ASUS\Downloads\market.csv", header=None)
df.head()
# View the dataset


# In[10]:


df.fillna(0,inplace=True)
df.head()
#replaced the null values wiht zero to see nicer


# In[11]:


#Here, the data represents the items a customer bought at a particular time at grocery market. 
#Dataset contains 20 columns and 7501 rows where each row represents the names of items in each respective columns.


# In[12]:


#[7501 rows x 20 columns]


# We need a data in form of 'list' for using Aprori Algorithm.

# In[13]:


transactions = [] 
for i in range(0,len(df)):
    transactions.append([str(df.values[i,j]) for j in range(0,20) if str(df.values[i,j])!='0'])                    


# In[14]:


print(type(transactions))


# In[35]:


transactions[0]


# The goal here is to apply Apriori algorithm on the dataset.
# Rules of support, confidence and lift as described as follows; 
# 1)Support: It is calculated to check how much popular a given item is. 
# It is measured by the proportion of transactions in which an itemset appears. 
# For example, there are 100 people who bought something from market, 
# amoung those 100 people, there are 20 people who bought "X" product. 
# Therefore, the support of people who bought "X" product will be (20/100 = 20%).
# 2)Confidence: It is calculated to check how likely if item "X" is purchased when item Y is purchased. 
# This is measured by the proportion of transactions with item "X", in which item Y also appears. 
# 3)Lift: It is calculated to measure how likely item Y is purchased when item "X" is purchased, 
# while controlling for how popular item Y is. 

# Apriori Algorithm
# We have provided min_support, min_confidence, min_lift of sample-set and mi_length to find the rule.
# Min_support  = 5(5 times a day) * 7 (7 days a week) 35/ 7501 = 0.0045 (dataset is for one week period)
# Min_confidence = set it lower to get more relations between products,so selected confidence level is 0.25
# Min_lift = In order to get some relevant rules,setting min_lift to 3.
# Min_length = 2 (at least two products is taken in the rule)

# In[16]:


from apyori import apriori

rules = apriori(transactions,min_support=0.0045,min_confidence=0.25,min_lift=3,min_length=2)
results = list(rules)

# we convert the rules into a list because it is easier to see the results in list format.


# In[17]:


df_results = pd.DataFrame(results)


# In[18]:


print("There are {} Relation derived.".format(len(results)))


# In[19]:


#df_results.head()
df_results.sort_values(by=['support'],ascending=False)
#Sorted according to support value descending


# In[20]:


print(results[2])
#Print any item to see our rule


# Accodring to result above, we can say that pasta and escalope are bought frequently. 
# The support for pasta is 0.0058. The confidence for this rule is 0.37288. means that out of all 
# the transactions containing pasta, 37.28% of the transactions are likely to contain escalope as well. 
# Lastly, lift of 4.70 shows that the escalope is 4.70 more likely to be bought by the customers that buy pasta, 
# compared to its default sale. 

# In[21]:


for item in results:
    # first index of the inner list
    
    pair = item[0]
    items = [x for x in pair]
    print("Rule: " + items[0] + " => " + items[1])

    # second index of the inner list which gives support value
    print("Support: " + str(item[1]))

    # third index of the list located at 0th
    # of the third index of the inner list

    print("Confidence: " + str(item[2][0][2]))
    print("Lift: " + str(item[2][0][3]))
    print("************************************")

