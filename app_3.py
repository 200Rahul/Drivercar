#!/usr/bin/env python
# coding: utf-8

# In[2]:


#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import streamlit as st
from sklearn import *
import pickle


# In[2]:


df = pickle.load(open('data.pkl', 'rb'))
best_model = pickle.load(open('car_prediction.pkl', 'rb'))

st.title('Car Selling Price Predictor')

st.header('Fill the details to predict the Car price')

# brand_name - drop down
brand = st.selectbox('Brand', df['brand_name'].unique())
# car_name - drop down
car = st.selectbox('Car Name', df['car_name'].unique())
# Year - drop down
year = st.selectbox('Year', df['year'].unique())
# KM Driven - number input
km = st.number_input('KMs Driven')
# fuel - drop down
fuel = st.selectbox('Fuel', df['fuel'].unique())
# seller_type - drop down
seller_type = st.selectbox('Are you an Individual or a Dealer', df['seller_type'].unique())
# transmission - drop down
transmission = st.selectbox('Transmission', df['transmission'].unique())
# owner - drop down
owner = st.selectbox('Owner', df['owner'].unique())



if st.button('Predict Car Price') :
    test_data = np.array([brand, car, year, km, fuel, seller_type, transmission, owner])
    test_data = test_data.reshape([1,8])

    st.success(f"Rs. {best_model.predict(test_data)[0].round(2)}")


# In[ ]:





