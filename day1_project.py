#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


# # 1. 데이터 둘러 보기

# In[2]:


df = pd.read_csv('trip.csv')


# In[3]:


df.head()


# In[4]:


df.tail()


# In[5]:


df.shape


# In[6]:


df.info()


# In[7]:


df.describe()


# In[8]:


df.describe(include='object')


# ## = 데이터 둘러보기 결과 =
# - 행: 22,701개, 컬럼: 9개
# - fare_amoutn에 결측치가 있음
# - pickup과 dropoff time의 Dtype 확인 필요
# - 날짜 Dtype 변경 필요
# - passenger_count, trip_distance, fare_amont에 이상치 확인 필요
# - 문자열 확인 필요

# # 2. 결측치 검토
# - fare_amount에 3개의 결측치가 있으나 0.000132의 낮은 비율로 제거함

# In[9]:


df.isna().sum()


# In[10]:


df.isna().mean()


# In[11]:


df.dropna(subset=['fare_amount'], inplace=True)


# In[12]:


df.shape


# In[13]:


df.isna().any()


# # 3. 이상치 검토

# ### 3.1 passenger_count 검토
# - 택시는 운전기사 포함 총 5명 탑승으로 간주하여 최대 승객은 4명
# - 4명 이상의 승객은 이상치로 간주
# - 주행거리, 주행시간, 요금의 상관관계 분석이 목적이므로 이상치를 제거하지 않고 중간값으로 대체함

# In[14]:


df['passenger_count'].value_counts(ascending=False)


# In[15]:


sns.boxplot(df['passenger_count'])


# In[16]:


passenger_count_median = df['passenger_count'].median()
df['passenger_count'] = df['passenger_count'].apply(lambda x: passenger_count_median if x > 4 else x)


# In[17]:


df['passenger_count'].value_counts(ascending=False)


# ### 3.2 fare_amount 검토
# - fare_amount가 0보다 작은 행은 총 20개로 전체 데이터 개수에 비해 작아 삭제
# - 30 이상의 trip_distance의 개수와 fare_amount를 비교한 결과, fare_amount가 400, 999.99는 이상치로 제거

# In[18]:


df['fare_amount'].value_counts(ascending=False)


# In[19]:


df[df['fare_amount'] <= 0].shape


# In[20]:


df = df[df['fare_amount'] > 0]


# In[21]:


sns.boxplot(df['fare_amount'])


# In[22]:


df['fare_amount'].sort_values(ascending=False)


# In[23]:


df['trip_distance'].sort_values(ascending=False)


# In[24]:


df = df[df['fare_amount'] < 400]


# In[25]:


df['fare_amount'].sort_values()


# ### 3. trip_distance 검토
# - 거리가 0 이하 제거

# In[26]:


df[df['trip_distance'] <= 0].shape


# In[27]:


df = df[df['trip_distance'] > 0]


# In[28]:


df[df['trip_distance'] <= 0].shape


# # 4. 중복값 검토

# In[29]:


df.duplicated().sum()


# In[30]:


df = df.drop_duplicates()


# In[31]:


df.shape


# # 5. 컬럼명 변경

# In[32]:


df.rename({'tpep_pickup_datetime': 'pickup_time', 'tpep_dropoff_datetime': 'drop_time'}, axis=1, inplace=True)


# In[33]:


df.head(1)


# # 6. payment_metod
# - Credit Card와 Debit Card를 Card로 변경
# - None 값 제거

# In[34]:


df['payment_method'].value_counts()


# In[35]:


df['payment_method'] = df['payment_method'].str.split(expand=True)[0].apply(lambda x: 'Card' if x == 'Debit' or x == 'Credit' else x)       


# In[36]:


df['payment_method'].value_counts()


# # 7. pickup_time과 drop_time의 타입 변경

# In[38]:


df['pickup_time'] = pd.to_datetime(df['pickup_time'])
df['drop_time'] = pd.to_datetime(df['drop_time'])


# In[39]:


df.info()


# # 8. 주행시간 계산
# - driving_minutes 컬럼 추가

# In[44]:


df['driving_minutes'] = (df['drop_time'] - df['pickup_time']).dt.total_seconds() / 60


# In[46]:


df.head()


# # 9. 주행시간, 주행거리, 요금의 상관관계

# In[47]:


df['total_fare'] = df['fare_amount'] + df['tolls_amount']


# In[50]:


correlation_matrix = df[['driving_minutes', 'trip_distance', 'total_fare']].corr()
correlation_matrix


# In[51]:


plt.figure(figsize=(8, 6))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1, square=True, linewidths=.5)
plt.title('Correlation Matrix Heatmap')
plt.show()


# In[ ]:




