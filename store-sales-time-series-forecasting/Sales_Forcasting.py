#!/usr/bin/env python
# coding: utf-8

# In[1]:


#import libraries here 
# Standard imports 
import os
import pandas as pd
import numpy as np
import autoviz
from dataprep.eda import create_report
from autoviz.AutoViz_Class import AutoViz_Class
import datetime as dt


# In[2]:


#Change the location to the folder
loc = 'C:\\Users\\sumit\\OneDrive\\My Drive\\IPBA course\\GitHub\\Kaggle-Time-series\\store-sales-time-series-forecasting'
os.chdir(loc)


# In[3]:


#Lets read some files
TrainingData = pd.read_csv('train.csv')


# In[4]:


create_report(TrainingData)


# In[5]:


AV = AutoViz_Class()
df_av = AV.AutoViz('train.csv')


# In[23]:


Test_read = pd.read_csv('test.csv')
Test_read['date'] = pd.to_datetime(Test_read['date'])
Test_read['date']=Test_read['date'].map(dt.datetime.toordinal)


# In[7]:


TrainingData.head()


# In[8]:


Test_read.head()


# In[4]:


TrainingData.dtypes


# In[5]:


TrainingData


# In[6]:


TrainingData_temp = TrainingData
TrainingData_temp['date'] = pd.to_datetime(TrainingData_temp['date'])
TrainingData_temp['date']=TrainingData_temp['date'].map(dt.datetime.toordinal)


# In[7]:


TrainingData_temp.head()


# In[8]:


TrainingData_temp.dtypes


# In[ ]:


Train


# In[29]:


#X = TrainingData_temp.drop(["id","sales"],axis=1)
X = TrainingData_temp.drop(["sales"],axis=1)
Y = TrainingData_temp['sales']


# In[30]:


X = pd.get_dummies(X)


# In[31]:


from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.3, random_state = 200)


# In[17]:


X.dtypes


# In[ ]:





# In[13]:


from sklearn.ensemble import RandomForestRegressor


# In[18]:


regressor = RandomForestRegressor(n_estimators=100,max_depth=3, random_state = 400, n_jobs=-1 )
regressor.fit(X_train, Y_train)
regressor.score(X_test, Y_test)


# In[32]:


regressor = RandomForestRegressor(n_estimators=100,max_depth=5, random_state = 400, n_jobs=-1 )
regressor.fit(X_train, Y_train)
regressor.score(X_test, Y_test)


# In[90]:


regressor = RandomForestRegressor(n_estimators=400,max_depth=3, random_state = 400, n_jobs=-1 )
regressor.fit(X_train, Y_train)
regressor.score(X_test, Y_test)


# In[ ]:





# In[33]:


pd.Series(regressor.feature_importances_,index=X.columns).sort_values(ascending=False)


# In[24]:


Test_read.head()


# In[26]:



X_TEST_Dummy= pd.get_dummies(Test_read)


# In[ ]:





# In[35]:


PredictedValues = regressor.predict(X_TEST_Dummy)


# In[36]:


PredictedValues


# In[37]:


Test_read["sales"]  = PredictedValues


# In[38]:


print(Test_read.dtypes)
print(Test_read.head())


# In[42]:


PredictedDataFrame = Test_read[['id','sales']]
PredictedDataFrame.to_csv('predicted.csv',encoding='utf-8')


# In[ ]:




