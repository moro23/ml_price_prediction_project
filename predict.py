#!/usr/bin/env python
# coding: utf-8

# In[1]:


## for data preprocessing
import numpy as np
import pandas as pd

## for data visualization
import matplotlib.pyplot as plt
import seaborn as sns

## split data into train and validation set
from sklearn.model_selection import train_test_split

## linear regression model 
from sklearn.linear_model import LinearRegression, Ridge
## 
from sklearn.feature_extraction import DictVectorizer

##
from sklearn.metrics import r2_score

## hyper paramter tuning
from sklearn.model_selection import RepeatedKFold
from sklearn.model_selection import RandomizedSearchCV

from scipy.stats import loguniform




# In[3]:


## read dataset
data = pd.read_csv('dataset/used_device_data.csv')


# In[4]:


## create a copy of the dataset
df = data.copy()


# In[18]:


df_train_full, df_test = train_test_split(df, test_size=0.2, random_state=11)
df_train, df_valid = train_test_split(df_train_full, test_size=0.25, random_state=11)

print(f'The Size Of The Training Set: {len(df_train)}')
print(f'The Size Of The Validation Set: {len(df_valid)}')
print(f'The Size Of The Test Set: {len(df_test)}')
print(f'The Size Of The Full Training Set: {len(df_train_full)}')


# In[19]:


df_train_full = df_train_full.fillna(0)
df_train = df_train.fillna(0)
df_valid = df_valid.fillna(0)
df_test = df_test.fillna(0)


# In[29]:


numerical_var = ['new_price', 'selfie_camera_mp', 'battery', 'ram', 'screen_size', 'release_year', 'main_camera_mp']

categorical_var = ['4g', '5g']


# In[40]:


y_train_full = np.log1p(df_train_full['used_price'])
y_test = np.log1p(df_test['used_price'])


# In[41]:


del df_train_full['used_price']
del df_test['used_price']


# In[42]:


df_train_full = df_train_full[numerical_var + categorical_var]
df_test = df_test[numerical_var + categorical_var]

train_full_dict = df_train_full.to_dict(orient='records')
train_test_dict = df_test.to_dict(orient='records')

dv = DictVectorizer(sparse=False)

X_train_full = dv.fit_transform(train_full_dict)
X_test = dv.transform(train_test_dict)


# In[43]:


rid_reg_final_model = Ridge(alpha=0.0001)
rid_reg_final_model.fit(X_train_full, y_train_full)


# In[44]:


test_score = rid_reg_final_model.score(X_test, y_test)
print(f'Test Accuracy: {round(test_score, 2)}')


# In[49]:


import bentoml

saved_model = bentoml.sklearn.save_model("used_price_prediction", rid_reg_final_model, custom_objects={"dictVectorizer": dv})
print(f"Model saved: {saved_model}")


# In[48]:


model = bentoml.sklearn.load_model("used_price_prediction:latest")

