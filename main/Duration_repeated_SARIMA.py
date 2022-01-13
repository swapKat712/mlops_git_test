#!/usr/bin/env python
# coding: utf-8

# # Astro PoC (repeated duration)

# In[1]:


import numpy as np
import pandas as pd
import os
from dateutil.parser import parse
import matplotlib.pyplot as plt

from statsmodels.tsa.seasonal import seasonal_decompose
import statsmodels.api as sm
from pmdarima import auto_arima
from statsmodels.tsa.statespace.sarimax import SARIMAX
### Testing For Stationarity
from statsmodels.tsa.stattools import adfuller

import datetime
import time


# In[2]:


import warnings
warnings.filterwarnings('ignore')


# In[3]:


df = pd.read_csv('merged_data.csv')


# In[4]:


def datefxn(date):
    """
    This function will convert string dtype date info to datetime dtype
    
    :param date: string dtype date
    :return: datetime dtype date
    """
    dt = parse(date)
    dt.strftime('%d/%m/%Y')
    return dt


# In[5]:


df['Date']=df.apply(lambda x: datefxn(x['date']), axis=1)
df.drop( ['date'], axis=1, inplace=True)
df.sort_values(['Date'], ascending=True, inplace=True)
df.reset_index(drop=True,inplace=True)


# In[6]:


# creating df_movies DataFrame with entries having at least 1 hour duration i.e. only movies content
df_movies = df[pd.to_datetime(df['Duration'], format='%H:%M:%S:%f').dt.hour!=0]


# In[7]:


showset=set()


# In[8]:


def newcol(showset,showname):
    if showname in showset:
        return 1     #old show
    else:
        showset.add(showname)
        return 0     #new show


# In[9]:


# 0 means new show
df_movies['newnotnew']=df_movies.apply(lambda x: newcol(showset,x['Title']), axis=1)


# In[10]:


def timefxn(duration):
    """
    This function will convert string dtype duration info to float dtype hourly duration
    
    :param duration: string dtype duration
    :return: float dtype duration in hours
    """
    x = time.strptime(duration.split(',')[0],'%H:%M:%S:%f')
    return datetime.timedelta(hours=x.tm_hour,minutes=x.tm_min,seconds=x.tm_sec).total_seconds()/3600


# In[11]:


df_movies['Duration_in_hr']=df_movies.apply(lambda x: timefxn(x['Duration']), axis=1)
df_movies.drop( ['Duration'], axis=1, inplace=True)


# In[12]:


# storing hourly duration of repeated content for each day in x
x = pd.DataFrame(df_movies[df_movies['newnotnew']==1].groupby(['Date','dayofweek'])['Duration_in_hr'].sum())
# storing hourly duration of total content for each day in merged_df
merged_df = pd.DataFrame(df_movies.groupby(['Date','dayofweek'])['Duration_in_hr'].sum())
# creating a column "duration_repeated" to store the duration of repeated content for each day (i.e. x)
merged_df['duration_repeated']=x
merged_df['%repeatedcontent'] = merged_df['duration_repeated']/merged_df['Duration_in_hr']
merged_df.reset_index(inplace=True)


# ## Set Rolling window size and cap value

# In[14]:


#Rolling window size (moving average)
roll_size = 3


# In[15]:


# new column "y" created to store moving averaged values
merged_df['y'] = merged_df['duration_repeated'].rolling(window=roll_size).mean()


# In[16]:


# creating train and test set
train=merged_df[merged_df['Date']<='2021-09-30']
test=merged_df[merged_df['Date']>'2021-09-30']


# # SARIMAX on MA data

# In[17]:


max_value = 24


# In[18]:


MA_data = merged_df[['Date','y']]


# In[19]:


MA_data.set_index('Date',inplace=True)
MA_data.dropna(inplace=True)


# In[20]:


MA_train = MA_data.loc['2021-08-03':'2021-09-30']
MA_test = MA_data.loc['2021-10-01':]


# # auto_arima - to find the best model (minimum AIC value)

# In[21]:


MA_stepwise_model = auto_arima(MA_data, start_p=1, start_q=1,
                           max_p=3, max_q=3, m=30,
                           start_P=0, seasonal=True,
                           d=1, D=1, trace=True,
                           error_action='ignore',  
                           suppress_warnings=True, 
                           stepwise=True)
# print(MA_stepwise_model.aic())


# In[22]:


print(MA_stepwise_model.summary())


# In[23]:


future_forecast = MA_stepwise_model.predict(n_periods=31)


# In[24]:


future_forecast_df = pd.DataFrame(future_forecast,index = MA_test.index,columns=['Prediction'])


# In[25]:


future_forecast_df[future_forecast_df['Prediction']>max_value]=max_value


# In[26]:


print(future_forecast_df)


# In[27]:


future_forecast_df.to_csv('future_forecast.csv')

