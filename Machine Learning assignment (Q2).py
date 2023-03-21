#!/usr/bin/env python
# coding: utf-8

# In[5]:


import pandas as pd
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cluster import KMeans
import numpy as np


# In[6]:


# Data cleaning & getting rid of irrelevant information before clustering
df = pd.read_csv('https://raw.githubusercontent.com/edyoda/data-science-complete-tutorial/master/Data/house_rental_data.csv.txt')
df


# In[7]:


df.info()


# In[8]:


df.drop('Unnamed: 0', axis=1, inplace=True)


# In[9]:


df.isnull().sum()


# In[10]:


df.describe()


# In[11]:


#Finding the optimal value of k

df1 = df[['Sqft', 'Price']]
df1


# In[12]:


list1=[]

for i in range(1,8):
   model= KMeans(n_clusters= i, init='k-means++', random_state=0)
   model.fit(df)
   list1.append(model.inertia_)


# In[13]:


list1


# In[14]:


plt.plot(range(1,8), list1)
plt.show()


# In[15]:


#Storing cluster to which the house belongs along with the data

final_model = KMeans(n_clusters = 3,init = "k-means++",random_state = 0)#my model of clusters have 5
final_model.fit(df1)
    
    
df1['Sqrt_label'] = final_model.labels_


# In[16]:


df1


# In[17]:


px.scatter(df1,x ="Sqft",y = "Price",color = "Sqrt_label")


# In[59]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




