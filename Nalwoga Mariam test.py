#!/usr/bin/env python
# coding: utf-8

# # Question 1

# In[1]:


import numpy as np
import pandas as pd


# In[6]:


A=np.array([[1,0.2,0.5],[0.2,1,0.8],[0.5,0.8,1]])
print(A)


# In[7]:


#Transpose
matrix_T=(np.transpose(A))
print(matrix_T)


# In[8]:


# Determinant of matrix A
np.linalg.det(A)


# # Question 2

# In[ ]:


import pandas as pd


# In[19]:


#constructing a matrix as a dataframe
df=pd.DataFrame({'A':[1,0.2,0.5],
                'B':[0.2,1,0.8],
                'C':[0.5,0.8,1]})
df.head()


# # Question 3

# In[ ]:





# In[14]:


#standard deviation of data
dataset=[1,3,1,2,9,4,5,6,10,4]
print('Standard deviation:',np.std(dataset))


# # Question 4

# In[20]:


Question 4=input('The value of x is')
print('Question 4')


# # Question 5

# In[27]:


#loading the csv covid-19 dataset
df=pd.read_csv(r'C:\Users\Mariam\Desktop\WITI PROGRAM\DATA\test\COVID-19 Cases.csv')
print(df)
df.set_index(Date)# setting the date as index


# In[30]:


#view of the first 10 lines
result=df.head(10)
print(result)


# # Question 6

# In[33]:


df_results = df[(df.Difference >0) & (df.Case_Type == 'Confirmed') & (df.Country_Region == 'Italy')]#Italy dataframe


# In[35]:


print(df_results)


# In[40]:


df_results1= df[(df.Difference >0) & (df.Case_Type == 'Confirmed') & (df.Country_Region == 'Germany')]#Dataframe of Germany
print(df_results1)


# In[38]:


df_results.sort_values(by=['Case_Type'])#Check whether the results are sorted by cases & ascending order to well and accordingly


# In[41]:


df_results1 = df[(df.Difference >0) & (df.Case_Type == 'Confirmed') & (df.Country_Region == 'Germany')]
print(df_results1)


# # Question 7

# In[39]:


import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[44]:


df_results1.Difference.value_counts().plot(kind='hist')#Histogram of difference column of Germany


# # Question 8
# 

# In[48]:


df_results.describe()#computation on summary statistics of atleast three columns of germany data frame


# # Question 9

# In[51]:


df_results1.boxplot(by='Country_Region', column=['Difference'], grid=False)#box plot of Germany


# # Question 10

# In[53]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[54]:


Covid_dataset=pd.read_csv(r'C:\Users\Mariam\Desktop\WITI PROGRAM\DATA\test\COVID-19 Cases.csv')#importing csv file as new data frame


# In[55]:


print(Covid_dataset)


# # Question 11

# In[61]:


df[(df.Country_Region=='Germany')&(df.Case_Type=='Confirmed')]


# # Question 12

# In[63]:


df.plot.scatter(x='Country_Region',y='Cases')
plt.xlabel('Country')
plt.ylabel('Cases')
plt.title('Scatter Plot for Countries and the Covid_19 Cases')
plt.show()


# In[ ]:





# In[ ]:




