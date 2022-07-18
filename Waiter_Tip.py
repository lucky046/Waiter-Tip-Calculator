#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go


# In[2]:


data=pd.read_csv('waitertip.csv')
print(data.head())


# In[3]:


data.info()


# In[4]:


figure=px.scatter(data_frame=data,x= 'total_bill', y='tip', size='size',color='day', trendline='ols' )
figure.show()


# In[5]:


figure=px.scatter(data_frame=data,x= 'total_bill', y='tip', size='size',color='smoker', trendline='ols' )
figure.show()


# In[6]:


figure=px.scatter(data_frame=data,x= 'total_bill', y='tip', size='size',color='sex', trendline='ols' )
figure.show()


# In[7]:


figure=px.scatter(data_frame=data,x= 'total_bill', y='tip', size='size',color='time', trendline='ols' )
figure.show()


# In[8]:


figure=px.pie(data,values='tip', names='day', hole=0.4)
figure.show()


# In[9]:


figure=px.pie(data,values='tip', names='sex')
figure.show()


# In[10]:


figure=px.pie(data,values='tip', names='smoker')
figure.show()


# In[11]:


figure=px.pie(data,values='tip',names='time')
figure.show()


# In[12]:


data['sex']=data['sex'].map({'Male':1,'Female':0})
data['smoker']=data['smoker'].map({'Yes':1,'No':0})
data['day']=data['day'].map({'Thur':0,'Fri':1,'Sat':2,'Sun':3})
data['time']=data['time'].map({'Lunch':0,'Dinner':1})
data.head()


# In[13]:


x=np.array(data[['total_bill','sex','smoker','day','time','size']])
y=np.array(data['tip'])


# In[14]:


from sklearn.model_selection import train_test_split
xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=0.2, random_state=42)


# In[15]:


from sklearn.linear_model import LinearRegression
model = LinearRegression()
model.fit(xtrain, ytrain)


# In[16]:


#to know the predicted value of tip
x=np.array([[30,0,1,3,0,8]])
model.predict(x)


# In[ ]:




