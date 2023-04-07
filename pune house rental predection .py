#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib 
matplotlib.rcParams["figure.figsize"] = (20,10)


# In[2]:


df1=pd.read_csv('Pune_rent.csv.zip')
df1


# In[3]:


df1.property_type.describe()


# In[4]:


df2=df1.drop(['seller_type',"layout_type"],axis=1)
df2


# In[5]:


df2.isnull().sum()


# In[6]:


df3=df2.dropna()
df3.isnull().sum()


# In[7]:


df3.bathroom.unique()


# In[8]:


df3.groupby('bathroom')['bathroom'].agg('count')


# In[9]:


valid_bathrooms = ['1 bathrooms', '2 bathrooms', '3 bathrooms', '4 bathrooms', '5 bathrooms', '6 bathrooms']
mask = df3['bathroom'].isin(valid_bathrooms)
df_filtered = df3[mask]


# In[10]:


df_filtered.groupby('bathroom')['bathroom'].agg('count')


# In[11]:


df_filtered


# In[12]:


df_filtered['bathrooms']=df_filtered.bathroom.apply(lambda x:int(x.split(' ')[0]))


# In[13]:


df_filtered


# In[14]:


df_filtered=df_filtered.drop('bathroom',axis=1)


# In[15]:


df_filtered.area.unique()


# In[16]:


def is_float(x):
    try:
        float(x)
    except:
        return False
    return True


# In[17]:


df_filtered[~df_filtered.price.apply(is_float)]


# In[18]:


def remove_comma(x):
    tokens=x.split(',')
    if len(tokens)==2:
        return float(str(tokens[0])+str(tokens[1]))
    try:
        float(x)
    except:
        return None


# In[19]:


df4=df_filtered.copy()
df4['price']=df4['price'].apply(remove_comma)
df4


# In[20]:


df4.price.unique()


# In[21]:


len(df4.locality.unique())


# In[22]:


df4.locality=df4.locality.apply(lambda x: x.strip())
location_stats=df4.groupby('locality')['locality'].agg('count').sort_values(ascending=False)
location_stats.head(100)


# In[23]:


len(location_stats[location_stats<=10])


# In[24]:


loc_less_then_10=location_stats[location_stats<=10]


# In[25]:


df4.locality=df4.locality.apply(lambda x: 'other' if x in loc_less_then_10 else x)
len(df4.locality.unique())


# In[26]:


df4


# In[27]:


df4[df4.area/df4.bedroom<300]


# In[28]:


len(df4)


# In[29]:


df5=df4[~(df4.area/df4.bedroom<300)]
len(df5)


# In[30]:


df5['price_per_unit_area'] = df5['price'] / df5['area']
df5


# In[31]:


df5.price_per_unit_area.describe()


# In[32]:


def remove_ppua_outliers(df):
    df_out=pd.DataFrame()
    for Key,subdf in df.groupby('locality'):
        m=np.mean(subdf.price_per_unit_area)
        st=np.std(subdf.price_per_unit_area)
        reduced_df=subdf[(subdf.price_per_unit_area>(m-st))&(subdf.price_per_unit_area<=(m+st))]
        df_out=pd.concat([df_out,reduced_df],ignore_index=True)
    return df_out


# In[33]:


len(df5)


# In[34]:


df6=remove_ppua_outliers(df5)
len(df6)


# In[35]:


df6.head(10)


# In[36]:


def plot_scatter_chart(df,location):
   bhk2 = df[(df.locality==location) & (df.bedroom==2)]
   bhk3 = df[(df.locality==location) & (df.bedroom==3)]
   matplotlib.rcParams['figure.figsize'] = (15,10)
   plt.scatter(bhk2.area,bhk2.price,color='blue',label='2 BHK', s=50)
   plt.scatter(bhk3.area,bhk3.price,marker='+', color='green',label='3 BHK', s=50)
   plt.xlabel("Total Square Feet Area")
   plt.ylabel("Price (Lakh Indian Rupees)")
   plt.title(location)
   plt.legend()
   
plot_scatter_chart(df6,"Kothrud")


# In[37]:


def remov_bedroom_outliers(df):
    exclude_indices=np.array([])
    for location,location_df in df.groupby('locality'):
        bedroom_stats={}
        for bedroom,bedroom_df in location_df.groupby('bedroom'):
            bedroom_stats[bedroom]={
                'mean': np.mean(bedroom_df.price_per_unit_area),
                'std': np.std(bedroom_df.price_per_unit_area),
                'count': bedroom_df.shape[0]
            }
        for bedroom,bedroom_df in location_df.groupby('bedroom'):
            stats=bedroom_stats.get(bedroom-1)
            if stats and stats['count']>5:
                exclude_indices=np.append(exclude_indices,bhk_df[bhk_df.price_per_unit_area<(stats['mean'])].index.values)
        return df.drop(exclude_indices,axis='index')
df7 = remov_bedroom_outliers(df6)
len(df7)
            

    


# In[38]:


def plot_scatter_chart(df,location):
   bhk2 = df[(df.locality==location) & (df.bedroom==2)]
   bhk3 = df[(df.locality==location) & (df.bedroom==3)]
   matplotlib.rcParams['figure.figsize'] = (15,10)
   plt.scatter(bhk2.area,bhk2.price,color='blue',label='2 BHK', s=50)
   plt.scatter(bhk3.area,bhk3.price,marker='+', color='green',label='3 BHK', s=50)
   plt.xlabel("Total Square Feet Area")
   plt.ylabel("Price (Lakh Indian Rupees)")
   plt.title(location)
   plt.legend()
   
plot_scatter_chart(df7,"Wadgaon Sheri")


# In[39]:


plt.hist(df7.price_per_unit_area)
plt.xlabel('price_per_unit_area')
plt.ylabel('count')


# In[40]:


df7.bathrooms.unique()


# In[41]:


df7[df7.bathrooms>df7.bedroom+1]


# In[42]:


df8=df7[df7.bathrooms<df7.bedroom+1]
df8.shape


# In[43]:


df9=df8.drop(['price_per_unit_area'],axis=1)
df9


# In[44]:


dummies = pd.get_dummies(df9[['locality', 'property_type', 'furnish_type']])
dummies


# In[45]:


df10=pd.concat([df9,dummies],axis=1)
df10


# In[46]:


df11=df10.drop(['locality', 'property_type', 'furnish_type'],axis=1)


# In[58]:


x=df11.drop(['price'],axis=1)
x.head()


# In[48]:


y=df11.price
y.head()


# In[49]:


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=40)


# In[50]:


from sklearn.linear_model import LinearRegression
lr_model=LinearRegression()
lr_model.fit(x_train,y_train)
lr_model.score(x_test,y_test)


# In[51]:


from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import cross_val_score

c=ShuffleSplit(n_splits=5,test_size=0.2,random_state=10)

cross_val_score(LinearRegression(),x,y,cv=c)


# In[52]:


from sklearn.model_selection import GridSearchCV

from sklearn.linear_model import Lasso
from sklearn.tree import DecisionTreeRegressor

def find_best_estimator_using_GSCV(x,y):
    algos={
        'linear regression' : {
            'model' :LinearRegression(),
            'params' :{
                'fit_intercept': [True,False]
            }
        },
        'lasso' :{
            'model':Lasso(),
            'params':{
                'alpha': [1,2],
                'selection':['random','cyclic']
            }
        },
        'DecisionTreeRegressor':{
            'model':DecisionTreeRegressor(),
            'params':{
                'criterion' : ["mse", "friedman_mse"],
                'splitter' : ["best", "random"]
            }
        }
    }
    score=[]
    for algo_names,config in algos.items():
        c=ShuffleSplit(n_splits=5,test_size=0.2,random_state=10)
        gs=GridSearchCV(config['model'],config['params'],cv=c,return_train_score=False)
        gs.fit(x,y)
        score.append({
            'model': algo_names,
            'best_score' : gs.best_score_,
            'best_params': gs.best_params_
        })
    return pd.DataFrame(score,columns=['model','best_score','best_params'])
find_best_estimator_using_GSCV(x,y)


# In[54]:


DTR=DecisionTreeRegressor(criterion='friedman_mse', splitter= 'best')
DTR.fit(x_train,y_train)
DTR.score(x_test,y_test)


# In[89]:


def predict_price(locality, area, bath, bedroom, property_type, furnish_type):
    loc_index = np.where(x.columns == locality)[0][0]
    prop_type_index = np.where(x.columns == property_type)[0][0]
    furnish_type_index = np.where(x.columns == furnish_type)[0][0]
    
    X = np.zeros(len(x.columns))
    X[0] = area
    X[1] = bath
    X[2] = bedroom
    
    if loc_index >= 0:
        X[loc_index] = 1
    
    if prop_type_index >= 0:
        X[prop_type_index] = 1
        
    if furnish_type_index >= 0:
        X[furnish_type_index] = 1
    
    return DTR.predict([X])[0]


# In[90]:


predict_price('locality_Wadgaon Sheri',350,1,1,"property_type_Studio Apartment",'furnish_type_Unfurnished')


# In[91]:


predict_price('locality_Wagholi',1085,3,3,"property_type_Apartment",'furnish_type_Semi-Furnished')


# In[95]:


predict_price('locality_Agalambe',350,1,1,"property_type_Studio Apartment",'furnish_type_Unfurnished')


# In[ ]:


from sklearn.metrics import r2_score


