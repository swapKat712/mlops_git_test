#!/usr/bin/env python
# coding: utf-8

# # Astro PoC (repeated duration)

# In[1]:


import numpy as np
import pandas as pd
import os
from dateutil.parser import parse
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import mean_absolute_percentage_error
from sklearn.metrics import mean_absolute_error

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


df2=pd.DataFrame(columns=['Time','Duration','Title','dayofweek','date'])
path_to_folders = './CINEMAX/CINEMAX'

for folder in os.listdir(path_to_folders):
    for root,dirs,files in os.walk(path_to_folders+'/'+folder):
        for file in files:
            if file.endswith(".xls"):
                cinemaxdf=pd.read_excel(os.path.join(path_to_folders+'/'+folder+'/'+file),header=None) # temp DataFrame
                date=cinemaxdf.iloc[0,0].split()[-3:] # extracting date info (last 3 entries of 1st line)
                date=" ".join(date) # joining last 3 entries of date into a single string
                dayofweek=cinemaxdf.iloc[0,0].split()[-4] # extracting day of week (4th entry from last of 1st line)
                cinemax=cinemaxdf.iloc[2:,:] # 2nd line is blank.. so considering entries from 3rd row onwards
                cinemax.columns=cinemax.iloc[0] # specifying column name for each column
                cinemax=cinemax[1:]
                cinemax.reset_index(drop=True, inplace=True)
                cinemax=cinemax.iloc[1:,:] # iloc[1:,:] as there is blank space after header containing column names
                cinemaxnew=cinemax[['Time','Duration','Title','Format Type']] # taking subset of columns
                cinemaxnew=cinemaxnew.loc[cinemaxnew['Format Type']=='XDCAM'] # considering only XDCAM entries
                cinemaxnew.reset_index(drop=True,inplace=True)
                cinemaxnew['dayofweek']=dayofweek
                cinemaxnew['date']=date
                cinemaxnew.drop( ['Format Type'], axis=1, inplace=True)
                df2=df2.append(cinemaxnew) # appending multiple entries for each day to df2
# df2


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


df2['Date']=df2.apply(lambda x: datefxn(x['date']), axis=1)
df2.drop( ['date'], axis=1, inplace=True)
df2.sort_values(['Date'], ascending=True, inplace=True)
df2.reset_index(drop=True,inplace=True)
# df2


# In[6]:


# creating df_movies DataFrame with entries having at least 1 hour duration i.e. only movies content
df_movies = df2[pd.to_datetime(df2['Duration'], format='%H:%M:%S:%f').dt.hour!=0]


# In[7]:


# df_movies.head()


# In[8]:


showset=set()


# In[9]:


def newcol(showset,showname):
    if showname in showset:
        return 1     #old show
    else:
        showset.add(showname)
        return 0     #new show


# In[10]:


df_movies['newnotnew']=df_movies.apply(lambda x: newcol(showset,x['Title']), axis=1)
# df_movies          # 0 means new show


# In[11]:


pd.set_option('display.max_rows', None)


# In[12]:


# df_movies


# In[13]:


def timefxn(duration):
    """
    This function will convert string dtype duration info to float dtype hourly duration
    
    :param duration: string dtype duration
    :return: float dtype duration in hours
    """
    x = time.strptime(duration.split(',')[0],'%H:%M:%S:%f')
    return datetime.timedelta(hours=x.tm_hour,minutes=x.tm_min,seconds=x.tm_sec).total_seconds()/3600


# In[14]:


df_movies['Duration_in_hr']=df_movies.apply(lambda x: timefxn(x['Duration']), axis=1)
df_movies.drop( ['Duration'], axis=1, inplace=True)
# df_movies


# In[15]:


# storing hourly duration of repeated content for each day in x
x = pd.DataFrame(df_movies[df_movies['newnotnew']==1].groupby(['Date','dayofweek'])['Duration_in_hr'].sum())
# storing hourly duration of total content for each day in merged_df
merged_df = pd.DataFrame(df_movies.groupby(['Date','dayofweek'])['Duration_in_hr'].sum())
# creating a column "duration_repeated" to store the duration of repeated content for each day (i.e. x)
merged_df['duration_repeated']=x
merged_df['%repeatedcontent'] = merged_df['duration_repeated']/merged_df['Duration_in_hr']
merged_df.reset_index(inplace=True)
# merged_df


# In[16]:


def onlydate(date):
    """
    This function will classify whether a date belongs to first half (return value of 1) or second half (return value of 2) of the month.
    
    :param date: datetime dtype date
    :return: 1 or 2
    """
    date=str(date)
    if date.strip()[8:10]<'15':
        return 1
    else:
        return 2


# In[17]:


# new column "half" indicating whether the date belongs to first or second half of the month.
merged_df['half']=merged_df.apply(lambda x: onlydate(x['Date']), axis=1)
# merged_df 


# ## Set Rolling window size and cap value

# In[18]:


#Rolling window size (moving average)
roll_size = 3


# In[19]:


# new column "y" created to store moving averaged values
merged_df['y'] = merged_df['duration_repeated'].rolling(window=roll_size).mean()


# In[20]:


# merged_df


# In[21]:


# creating train and test set
train=merged_df[merged_df['Date']<='2021-09-30']
test=merged_df[merged_df['Date']>'2021-09-30']


# # SARIMAX on MA data

# In[22]:


max_value = 24


# In[23]:


MA_data = merged_df[['Date','y']]
# MA_data.head()


# In[24]:


MA_data.set_index('Date',inplace=True)
MA_data.dropna(inplace=True)


# In[25]:


# MA_data.head()


# In[26]:


MA_train = MA_data.loc['2021-08-03':'2021-09-30']
MA_test = MA_data.loc['2021-10-01':]


# # auto_arima - to find the best model (minimum AIC value)

# In[27]:


MA_stepwise_model = auto_arima(MA_data, start_p=1, start_q=1,
                           max_p=3, max_q=3, m=30,
                           start_P=0, seasonal=True,
                           d=1, D=1, trace=True,
                           error_action='ignore',  
                           suppress_warnings=True, 
                           stepwise=True)
# print(MA_stepwise_model.aic())


# In[28]:


print(MA_stepwise_model.summary())


# In[29]:


# MA_stepwise_model.order


# In[30]:


# MA_stepwise_model.seasonal_order


# In[31]:


future_forecast = MA_stepwise_model.predict(n_periods=31)


# In[32]:


# future_forecast


# In[33]:


future_forecast_df = pd.DataFrame(future_forecast,index = MA_test.index,columns=['Prediction'])


# In[34]:


future_forecast_df[future_forecast_df['Prediction']>max_value]=max_value


# In[42]:


# future_forecast_df.head()


# In[36]:


#future_forecast = pd.DataFrame(future_forecast,index = test1.index,columns=['Prediction'])
pd.concat([MA_test,future_forecast_df],axis=1).plot()


# In[37]:


pd.concat([MA_data,future_forecast_df],axis=1).plot()


# In[38]:


print("MAPE on MA value: "+str(round(100*mean_absolute_percentage_error(MA_test['y'],future_forecast_df['Prediction']),2))+"%")


# In[39]:


print("MAE on MA value: "+str(round(mean_absolute_error(MA_test['y'],future_forecast_df['Prediction']),2))+" hours")


# In[40]:


print("MAPE on actual value: "+str(round(100*mean_absolute_percentage_error(test['duration_repeated'],
                                                                            future_forecast_df['Prediction']),2))+"%")


# In[41]:


print("MAE on actual value: "+str(round(mean_absolute_error(test['duration_repeated'],
                                                            future_forecast_df['Prediction']),2))+" hours")


# In[ ]:




