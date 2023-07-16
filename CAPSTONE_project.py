#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")


# In[2]:


df = pd.read_csv("CAR DETAILS.csv")   # importing the dataset
df.head()


# #### Checking columns name in dataset

# In[3]:


df.columns


# ### Shape of Dataset

# In[4]:


df.shape


# ### To see null values in dataset

# In[5]:


df.isnull().sum()


# ### Datatypes of columns

# In[6]:


df.dtypes


# ### See the unique values containing columns

# In[7]:


df[['fuel','seller_type','transmission','owner']].nunique()


# In[8]:


print(sorted(df.year.unique()))


# In[9]:


unique_name = df.name.unique()
print(unique_name.sum())


# In[10]:


df['fuel'].unique()


# In[11]:


print(df.seller_type.unique())


# In[12]:


df.transmission.unique()


# In[13]:


print(df.owner.unique())


# ### See statistics overview of dataset

# In[14]:


df.describe()


# In[15]:


df.describe(include='all')


# ## Checking the Duplicated values

# In[16]:


df.duplicated().sum()


# #### There are 763 duplicates values but we don't delete it from dataset because a single car is used many customer.

# #                      DATA ANALYSIS

# In[17]:


# Extracting brand name and car model name from name column

df['brand_name'] = df['name'].str.split().str[0]
df['car_name'] = df['name'].str.split().str[1]


# In[18]:


# Removing the name column

df.drop('name', axis=1, inplace=True)
df.head()


# In[19]:


# Moving the brand_name and car_name to the froont of the table

brand = df['brand_name']
car = df['car_name']
df.drop(['brand_name', 'car_name'], axis=1, inplace = True)
df.insert(0,'brand_name', brand)
df.insert(1, 'car_name', car)
df.head()


# In[20]:


df.columns


# In[21]:


df.info


# In[ ]:





# ### Let's start trying to find out What is the most sold car
# 

# In[22]:


df["brand_name"].value_counts(normalize = True)[:5].plot(kind = 'bar') 

plt.show() 


# ### We want to know the average prices of the brands

# In[23]:


#we use groupby and mean to extract values and plot to draw the graph

price = df.groupby([df["brand_name"]])[['selling_price']].mean()

price.sort_values(by='selling_price', ascending=True, inplace=True)

ax = price.plot(kind='barh', cmap='PRGn' , figsize=(10,16) ,title= 'Avarege Selling Price Car Brand')
for c in ax.containers: # set the bar label
      ax.bar_label(c, fmt='%.0f',label_type='center', color='w',rotation=0)


# ### We will display a graph to display sales

# In[24]:


## We will display sales by count in pie graphs

labels = df["brand_name"][:20].value_counts().index # We chose only twenty

sizes = df["brand_name"][:20].value_counts()

data = df.groupby(["brand_name"])["brand_name"].count().sort_values(ascending=False) # to extract the count

x = data.index # to extract the brand name
y = data.values # to extract the count to brand 

colors = ['#F8EEFB','#66b3ff','#8000FF','#ffcc99',"#00FF1B","#FF8040","#F8AEF8"]#color choice

plt.figure(figsize = (8,8)) # Determine the size of the graph

# Creating explode data

# explode = (0.1, 0.0, 0.2, 0.3, 0.0, 0.0)

plt.pie(sizes, labels=labels, rotatelabels=False, autopct='%1.1f%%',colors=colors,shadow=True, startangle=45)#Fomat pie
plt.title('name',color = 'black',fontsize = 15) # Fomat title

# plt.legend() <br>

plt.legend(title = "Cars") # title legend

# plt.legend(wedges, cars, title ="Cars",loc ="center left",bbox_to_anchor =(1, 0, 0.5, 1))
# myexplode = [0.2, 0, 0, 0]
plt.show() # view


# ### We see that from the above pie chart the most seller of car brand is "MARUTI" which the 35% persent of whole selling. This selling is about all the year so to see more information about selling we now check the year selling  

# ### See the yearwise selling count of all cars

# In[25]:


df.year.value_counts()


# In[26]:


# Here I am using another way to display the graph by seaborn we imported it in the beginning

sns.countplot(data=df,x="year")
plt.xticks(rotation=90)
plt.xlabel("YEAR",fontsize=10,color="forestgreen")
plt.ylabel("COUNT",fontsize=10,color="brown")
plt.title("YEAR COUNT")
plt.grid()
plt.show()


# ### Another graph to view sales classified by fuel
# ### We will display bar graphs
# 

# In[27]:


df["fuel"].value_counts(sort =True).plot(kind="bar", color=["green"])
plt.figsize=(8, 4) 


# ## Now see the seller view of sellertype by graph 

# In[28]:


df["seller_type"].value_counts(sort = True).plot(kind="bar", color=["green"], figsize=(8, 4) , title='Seller type')
plt.show()


# ### Now see the Transmission by graph

# In[29]:


df["transmission"].value_counts(sort = True).plot(kind="bar", color=["green"], figsize=(8, 4) , title='Transmission')
plt.show()


# In[30]:


df["owner"].value_counts(sort = True).plot(kind="bar", color=["green"], figsize=(8, 4) , title='Owner')
plt.show()


# In[31]:


def line_plot(data, title ,xlabel, ylabel):
    plt.figure(figsize=(8, 4))
    sns.lineplot(data=data , palette="tab10", linewidth=3.0)
    plt.title(title, fontsize=12)
    plt.ylabel(ylabel, size=14)
    plt.xlabel(xlabel, size=16)
    
df_price_move = df.groupby(['year'])[['selling_price']].mean()
line_plot(df_price_move,'Price Move', 'Year', "Price")


# #### We are able to see in line graph that as passes the year the price of cars was also increased but at last year it goes down

# ### This graph shows relations between price and km_driven a car

# In[32]:


df.plot(x="km_driven", y="selling_price", kind="scatter", figsize=(8, 4), title="Price & km_driven", color="green");


# ### The relationship between km_driven and year by scatter plot

# In[33]:


df.plot(x="year", y="km_driven", kind="scatter", figsize=(8, 4), title="Year", color="green");


# ## These are some different graphs are shown from above we can easily undestand about data now see the relation between different fators using correlation funtion

# In[34]:


df.corr()


# In[35]:


corr = df.corr()
corr = corr['selling_price']
corr = corr.sort_values(ascending=False)
sns.heatmap(df.corr(), annot=True)
plt.show()


# The heatmap shows a positive correlation between the "year" and "price" variables. As the year increases, the price tends to increase as well.This means that newer cars generally have higher prices compared to older cars.
# The intensity of the positive correlation is depicted by the color gradient in the heatmap, where darker shades indicate a stronger positive correlation.
# 
# Negative correlation between kilometers driven and price:
# 
# The heatmap reveals a negative correlation between the "kilometers driven" and "price" variables. When the number of kilometers driven increases, the price tends to decrease.
# This implies that cars with higher mileage are typically priced lower compared to cars with lower mileage. The intensity of the negative correlation is depicted by the color gradient in the heatmap, with darker shades indicating a stronger negative correlation.

# ## Now i use for loop 

# In[36]:


for i in df.year.unique():
    sns.lmplot(x='km_driven', y='selling_price', data=df[(df.year==i)])
    plt.ticklabel_format(style='plain', axis='y')
    plt.title(f'Year: {i}')


# #### The plot which i used in for loop is lmplot from seaborn and this all plot show that year wise car selling price and the car drive on that year

# In[ ]:





# In[ ]:





# In[37]:


from pycaret.regression import *


# In[38]:


# Initialize the regression setup
reg_setup = setup(data=df, target='selling_price', session_id=123)
# Compare models and select the best one
best_model = compare_models()
# Print the output metrics
print(best_model)


# In[ ]:





# In[39]:


x = df.drop('selling_price', axis=1)
y = df['selling_price']

print(type(x))
print(type(y))

print(x.shape)
print(y.shape)


# In[40]:


from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.25, random_state = 1)

print(x_train.shape)
print(x_test.shape)

print(y_train.shape)
print(y_test.shape)


# In[41]:


from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Function for Regression Evaluation Metrics

test = {'Model' : [], 'MAE' : [], 'MSE' : [], 'R2Score' : []}
def eval_model(model_name, y, ypred) :
    mae = mean_absolute_error(y, ypred)
    mse = mean_squared_error(y, ypred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y, ypred)
    print('Mean Absolute Error', mae)
    print('Mean Squared Error', mse)
    print('Root Mean Squared Error', rmse)
    print('R2 Score', r2)
    test['Model'].append(model_name)
    test['MAE'].append(mae)
    test['MSE'].append(mse)
    test['R2Score'].append(r2)


# In[ ]:





# In[42]:


from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge, Lasso
from sklearn.neighbors import KNeighborsRegressor 
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline


# In[43]:


x.head()


# ## 1) Lasso

# In[44]:


step1 = ColumnTransformer(transformers=
                         [('col_transf', OneHotEncoder(handle_unknown='ignore',drop = 'first', sparse=False), [0,1,4,5,6,7])],
                         remainder='passthrough')
step2 = Lasso(alpha=0.1)
pipe_lasso = Pipeline([('step1', step1), ('step2', step2)])
pipe_lasso.fit(x_train, y_train)

ypred_lasso = pipe_lasso.predict(x_test)

eval_model('Lasso Regression', y_test, ypred_lasso)


# ## 2) Linear Regression

# In[45]:


step1 = ColumnTransformer(transformers=
                         [('col_transf', OneHotEncoder(handle_unknown='ignore',drop = 'first', sparse=False), [0,1,4,5,6,7])],
                         remainder='passthrough')
step2 = LinearRegression()
pipe_lr = Pipeline([('step1', step1), ('step2', step2)])
pipe_lr.fit(x_train, y_train)

ypred_lr = pipe_lr.predict(x_test)

eval_model('Linear Regression', y_test, ypred_lr)


# ## 3) Ridge

# In[46]:


step1 = ColumnTransformer(transformers=
                         [('col_transf', OneHotEncoder(handle_unknown='ignore',drop = 'first', sparse=False), [0,1,4,5,6,7])],
                         remainder='passthrough')
step2 = Ridge(alpha=10)
pipe_ridge = Pipeline([('step1', step1), ('step2', step2)])
pipe_ridge.fit(x_train, y_train)

ypred_ridge = pipe_ridge.predict(x_test)

eval_model('Ridge Regression', y_test, ypred_ridge)


# ## 4) Decision Tree Regressor

# In[47]:


step1 = ColumnTransformer(transformers=
                         [('col_transf', OneHotEncoder(handle_unknown='ignore',drop = 'first', sparse=False), [0,1,4,5,6,7])],
                         remainder='passthrough')

step2 = DecisionTreeRegressor(max_depth=8, min_samples_split=15)
pipe_dt = Pipeline([('step1', step1), ('step2', step2)])
pipe_dt.fit(x_train, y_train)

ypred_dt = pipe_dt.predict(x_test)

eval_model('Decision Tree Regressor', y_test, ypred_dt)


# ## 5) Random Forest Regressor

# In[48]:


step1 = ColumnTransformer(transformers=
                         [('col_transf', OneHotEncoder(handle_unknown='ignore',drop = 'first', sparse=False), [0,1,4,5,6,7])],
                         remainder='passthrough')

step2 = RandomForestRegressor(n_estimators=100,max_depth=16, min_samples_split=15, random_state=11)
pipe_rf = Pipeline([('step1', step1), ('step2', step2)])
pipe_rf.fit(x_train, y_train)

ypred_rf = pipe_rf.predict(x_test)

eval_model('Random Forest Regressor', y_test, ypred_rf)


# In[ ]:





# ## 6) Extra Trees Regressor

# In[49]:


from sklearn.tree import ExtraTreeRegressor


# In[50]:


from sklearn.ensemble import ExtraTreesRegressor

step1 = ColumnTransformer(transformers=
                         [('col_transf', OneHotEncoder(handle_unknown='ignore',drop = 'first', sparse=False), [0,1,4,5,6,7])],
                         remainder='passthrough')

step2 = ExtraTreesRegressor(n_estimators=100,max_depth=16, min_samples_split=20, random_state=25)
pipe_rf = Pipeline([('step1', step1), ('step2', step2)])
pipe_rf.fit(x_train, y_train)

ypred_rf = pipe_rf.predict(x_test)

eval_model('Extra Trees Regressor', y_test, ypred_rf)


# #### i used 7 different - different ML model to see the which provide the high accuracy so that the random dorest regressor and Extra trees regressor both provide estimatly nearly accuracy so that i choose random forest regressor for the best model and save this model

# #  Save The Model

# In[51]:


import pickle


# In[52]:


pickle.dump(pipe_rf, open('car_prediction.pkl', 'wb'))    # Saving the best performing model
pickle.dump(df, open('data.pkl', 'wb'))  


# # Load the Model

# In[53]:


loaded_model = pickle.load(open('car_prediction.pkl', 'rb'))  


# ## Performing the model on 20 randomly selected data points 

# In[54]:


df_random_sample =df.sample(20)
df_random_sample.head()


# In[55]:


df_sample =df_random_sample.to_csv("car_details_sample_data")


# In[56]:


df_random_sample.shape


# In[57]:


new_x = df_random_sample.drop('selling_price', axis=1)
new_ytest =df_random_sample['selling_price']
print(new_x.shape)

best_pred = loaded_model.predict(new_x)
print(eval_model('Best Model', new_ytest, best_pred))
best_pred


# ### The accuracy of sample data is good after choosing 20 sample point from dataset and it is best model for it.

# In[ ]:




