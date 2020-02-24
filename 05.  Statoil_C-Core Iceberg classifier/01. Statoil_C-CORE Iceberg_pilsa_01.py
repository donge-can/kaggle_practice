#!/usr/bin/env python
# coding: utf-8

# 원본 커널 : https://www.kaggle.com/devm2024/keras-model-for-beginners-0-210-on-lb-eda-r-d
# 
# 위성으로 찍은 사진이 빙하인지 선박인지 구분해 내는 모델을 만드는 것

# In[1]:


import numpy as np
import pandas as pd

from subprocess import check_output


from sklearn.model_selection import train_test_split
from os.path import join as opj
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import pylab
plt.rcParams['figure.figsize'] = 10, 10
get_ipython().run_line_magic('matplotlib', 'inline')
# print(check_output(['ls', '']))


# In[2]:


path = 'C:/Users/user/Desktop/kaggle_data/05. Statoil_C-CORE_Iceberg/'


# In[3]:


train = pd.read_json(path+'input/train.json')


# In[4]:


test = pd.read_json(path+'input/test.json')


# <br>
# 
# ### Intro about the Data

# - 데이터 영상 정보
#     - 2채널로 이루어져 있음(`HH`, `HV`)
#     - `HH` : transmit / receive horizontally
#     - `HV` : transmit horizontally and receive vertically
#     - 보편적인 8비트 이미지(0~255)가 아니라 `소수점을 가지는 값`
#     
#     
# - 입사각 `incidence angle`
#     - 입사각이 똑같은 관측치가 존재함
#     - 입사각이 같은 데이터는 대부분 같은 범주에 속함
#     
# - 데이터 정제
#     - 같은 입사각을 가지는 군집은 모두 같은 범주에 속한다는 가정하에, 하나의 군집 안에서 극소수 데이터들이 대다수의 범주와 다를 경우, 소수의 범주를 대다수 범주로 수정

# In[5]:


train[['is_iceberg']].head()


# In[6]:


train.head()


# In[7]:


len(train.band_1[0])
# 1차원 배열로 이루어져 있음


# In[8]:


75 * 75


# <br>
# 
# ### Create 3 bands having HH, HV and avg of both

# In[9]:


# 2차원 배열로 변경
X_band_1 = np.array([np.array(band).astype(np.float32).reshape(75, 75) for band in train['band_1']])
X_band_2 = np.array([np.array(band).astype(np.float32).reshape(75, 75) for band in train['band_2']])
X_train = np.concatenate([X_band_1[:, :, :, np.newaxis], X_band_2[:, :, :, np.newaxis], ((X_band_1+X_band_2) /2)[:, :, :, np.newaxis]], axis = -1)


# In[17]:


X_train[:1]


# In[10]:


import plotly.offline as py
import plotly.graph_objs as go
py.init_notebook_mode(connected=True)
def plotmy3d(c, name):

    data = [
        go.Surface(
            z=c
        )
    ]
    layout = go.Layout(
        title=name,
        autosize=False,
        width=700,
        height=700,
        margin=dict(
            l=65,
            r=50,
            b=65,
            t=90
        )
    )
    fig = go.Figure(data=data, layout=layout)
    py.iplot(fig)
plotmy3d(X_band_1[12,:,:], 'iceberg')


# In[11]:


plotmy3d(X_band_1[14, :, :], 'Ship')


# In[ ]:


from matplotlib import pyplot

from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential

from keras.layers import Conv2D, MaxPooling2D, Dense, Dropout, Input, Flatten, Activation
from keras.layers import GlobalMaxPooling2D
from keras.layers.normalization import BatchNormaliztion
from keras.layers.merge import Concatenate

from keras.models import Model


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




