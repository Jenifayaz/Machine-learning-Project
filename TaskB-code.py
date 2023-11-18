#!/usr/bin/env python
# coding: utf-8

# # Part A

# ### Importing the required libraries

# In[1]:


get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
import seaborn as sns

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PowerTransformer

import math
import scipy.stats as stats
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.stats.stattools import durbin_watson

from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso
from sklearn.svm import SVR
from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline

from sklearn.metrics import mean_squared_error, r2_score


# ### Loading the Dataset
# 
# **Read the dataset into Pandas DataFrame and storing it in a variable data.**

# In[131]:


data = pd.read_csv("CE802_P3_Data.csv")
data.head()


# ### Analysing the Data

# In[65]:


print(f"The dataset has {data.shape[0]} rows and {data.shape[1]} columns")


# In[66]:


data.describe()


# **Analysing the datatype for each of the features present in the Dataframe.** 

# In[67]:


data.dtypes


# **The features 'F6' and 'F9' has the datatype as object.**
# 
# **Analysing the features 'F6' and 'F9' and displaying the unique values present in the features.**

# In[68]:


print("Unique lables in 'F6' : ",data['F6'].unique())
print("Unique labels in 'F9' : ",data['F9'].unique())


# In[69]:


print("Number of samples present for each categorical label from 'F6' : ")
print("\n",data['F6'].value_counts())
print("\nNumber of samples present for each categorical label from 'F9' : ")
print("\n",data['F9'].value_counts())


# **Checking for any NaN values in the dataset.**

# In[70]:


for col in data.columns:
    print('\t%s: %d' % (col,data[col].isna().sum()))


# **There are no missing values in the dataset.**

# In[71]:


sns.pairplot(data= data, diag_kind = 'kde')


# **The scatter plot shows that most of the features are normally distributed. Feature 'F4' is left-skewed and 'F23' is right-skewed.**

# In[132]:


data_1 = data.copy() # Working with the copy of the data


# ### Encoding data
# 
# **Converting the features having categorical labels to numerical labels by doing One hot encoding.** 

# In[133]:


def encoding(data):
    data_en = pd.get_dummies(data, drop_first = True)
    return data_en


# In[134]:


new_data = encoding(data_1)
new_data


# In[79]:


independent_variable = new_data.drop(['Target'], axis = 1)
dependent_variable = new_data['Target']


# ### Multicolinearity
# 
# **Exisitance of high correlation between independent variables is Multicolinearity. Presence of Multi Colinearity can destabilize the regression model. So it's important to identify and remove them if any present in the data.**

# In[80]:


plt.figure(figsize=(40,30))
sns.heatmap(independent_variable.corr(), annot  = True)


# **Using variance inflation factor to remove highly correlated variables. The variance inflation factor is a measure for the increase of the variance of the parameter estimates.**
# 
# **Will be removing variables with more than the value of 5.**

# In[81]:


data_before = independent_variable

x1 = sm.tools.add_constant(data_before)

series_before = pd.Series([variance_inflation_factor(x1.values , i) for i in range(x1.shape[1])], index = x1.columns)

display(series_before)


# **No features have value more than 5. So the variables are not correlated. Hence there is no Multicolinearity present in the data.**

# ### Data Transformation
# 
# **By analysing the data, few features are not normally distributed. As the Linear Regression assumes that the features are normally distributed, it is important to do Transformation on data for better performance of the model.**
# 
# **Using Q-Q plot to check whether the data is normally distributed or not.**

# In[135]:


def diagnosticplot(data,variable):
    
    plt.figure(figsize=(15,6))
    plt.subplot(1,2,1)
    data[variable].hist()
    
    plt.subplot(1,2,2)
    stats.probplot(data[variable], dist = 'norm', plot = plt)
    
    plt.show()


# In[136]:


diagnosticplot(new_data, 'F4')


# **From the above Q-Q plot, for the feature 'F4' all the data points are not in the same point. Also from the histogram, it can be understood as the data in 'F4' is left skewed. So Transformation can be applied to this feature. As the samples have negative values, feature 'F4' is transformed using Power Transformer.**

# In[137]:


scaler = PowerTransformer(method = 'yeo-johnson')

reshaped_data = np.array(new_data['F4']).reshape(-1, 1)
new_data['pt_F4'] = scaler.fit_transform(reshaped_data)


# **Creating new feature 'pt_F4' for storing the transformed values in it and droping the feature 'F4' from the data.**

# In[138]:


new_data = new_data.drop(['F4'],axis = 1)
new_data.shape


# **The below plot shows the skewed data is transformed and it is normally distributed.**

# In[139]:


diagnosticplot(new_data, 'pt_F4')


# **Analysing the feature 'F23' and it's also not normally distributed. This is right skewed and the data points are not in the line.** 

# In[140]:


diagnosticplot(new_data, 'F23')


# In[141]:


#Square Root Transformation
new_data['srt_f23'] = new_data['F23']**(1/2)
diagnosticplot(new_data, 'srt_f23')


# In[142]:


new_data = new_data.drop(['F23'],axis = 1)
new_data.shape


# **Independent  variable 'X' and Dependent variable 'y'**

# In[146]:


X = new_data.drop(['Target'], axis = 1)
y = new_data['Target']

print("The shape of 'X' containing the predictors is:", X.shape)
print("The shape of 'y' containing the label is:",y.shape)


# ### Splitting the Data for Training and Validation
# 
# **The dataset is split for training and validating data. The training dataset will be used for training and fine tuning the model and the validation dataset will be used for validating the model. After experimenting with different splits, training the model with more data improved the performance of the model. So the training dataset will be having 90% of the data and the remaining 10% of the data for validation dataset.**

# In[148]:


X_train, X_val,y_train, y_val = train_test_split(X,y, test_size = 0.1, random_state = 0)
print(X_train.shape)
print(y_train.shape)
print(X_val.shape)
print(y_val.shape)


# ### Models
# 
# **After experimenting with number of Machine learning algorithms, the selected best models for this dataset are,**
# 
# **1.Linear Regression**
# 
# **2.Lasso Regression**
# 
# **3.Support Vector Regressor (SVR)**

# ### Feature Scaling
# 
# **Standardizing the data using StandardScaler before training and validating the model.** 
# 
# **Standardization rescales the data to have mean of 0 and standard deviation of 1.**
# 
# **It is calculated by,**
# 
#                     X_new = (X-mean)/std

# In[149]:


from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.fit_transform(X_val)


# ### Model Training and Validation
# 
# ### Linear Regression

# In[150]:


model = LinearRegression(normalize = True)
model.fit(X_train,y_train)


# In[151]:


y_pred_lr1 = model.predict(X_val)
 
print("Training Accuracy for Linear regression model : ", model.score(X_train,y_train))
print("Validation Accuracy for Linear regression : ",model.score(X_val,y_val))


# In[152]:


acc_lr = round(r2_score(y_val,y_pred_lr1),2)
acc_lr


# In[153]:


print("Coefficients : ",model.coef_)
print("Intercept : ",model.intercept_)
print("Mean Squared Error (MSE) : %.2f" % mean_squared_error(y_val,y_pred_lr1))
mse = mean_squared_error(y_val,y_pred_lr1)
print("Root Mean Squared Error (RMSE) : ", math.sqrt(mse))
print("Coefficient of Determination (R^2) : %.2f" % r2_score(y_val,y_pred_lr1))


# ### Performing grid search to find tuned parameters for SVR model and Lasso Regression.

# In[154]:


def gridsearchcv(X,y):
    algorithms = {
       
        'lasso' : {
            'model' : Lasso(),
            'params' : {
                'alpha' : [1,2],
                'selection' : ['random', 'cyclic']
            }
        },
        'SVR' : {
            'model' : SVR(),
            'params' : {
                "C":[1,10,100,1000],
                "gamma" : [0.1,0.01,0.001,0.00001],
                "kernel":["rbf","poly","linear"]
            }
        }
    }
    
    scores = []
    cv = ShuffleSplit(n_splits = 5, test_size = 0.2, random_state = 0)
    for algo, config in algorithms.items():
        gridsearch = GridSearchCV(config['model'], config['params'], cv=cv, return_train_score = False)
        gridsearch.fit(X,y)
        scores.append({
            'models' : algo,
            'best_score' : gridsearch.best_score_,
            'best_params' : gridsearch.best_params_
        })
        
    return pd.DataFrame(scores, columns = ['models','best_score','best_params'])


# In[155]:


gridsearchcv(X_train,y_train)


# ### Lasso model Training and Validation

# In[156]:


l_regressor = Lasso(alpha = 2, selection = 'cyclic', normalize = True)
l_regressor.fit(X_train,y_train)


# In[157]:


y_pred = l_regressor.predict(X_val)
print("Training Accuracy for Lasso model : ",l_regressor.score(X_train,y_train))
print("Validation Accuracy for Lasso model : ",l_regressor.score(X_val,y_val))


# In[158]:


acc_lass0 = l_regressor.score(X_val,y_val)


# ### SVR model Training and Validation

# In[159]:


regressor = SVR(C = 1000, gamma = 0.01, kernel = 'rbf')
regressor.fit(X_train, y_train)


# In[160]:


print("Training Accuracy for SVR model : ",regressor.score(X_train,y_train))
print("Validation Accuracy for SVR model : ",regressor.score(X_val, y_val))


# In[161]:


acc_svr = regressor.score(X_val, y_val)
acc_svr


# In[162]:


models = ["Linear Regression", "Lasso Regression", "SVR"]
scores = []
scores.append(acc_lr)
scores.append(acc_lass0)
scores.append(acc_svr)

df = pd.DataFrame()

df['models'] = models
df['scores'] = scores 
df


# In[163]:


sns.set(style='whitegrid')
ax = sns.barplot(y='models', x='scores',data=df)


# **The barplot shows the Linear Regression outperforms all the other two models.**

# ### Part B

# In[171]:


# HERE YOU WILL USE THIS TEMPLATE TO SAVE THE PREDICTIONS ON THE TEST SET

# Load the test data
test_df = pd.read_csv('CE802_P3_Test.csv')
test_df


# In[183]:



# Make sure you work on a copy
test_data = test_df.iloc[:,:-1].copy()


# In[184]:


test_data = encoding(test_data)


# In[185]:


test_data


# In[175]:


diagnosticplot(test_data, 'F4')


# In[176]:


scaler = PowerTransformer(method = 'yeo-johnson')

reshaped_data = np.array(test_data['F4']).reshape(-1, 1)
test_data['pt_F4'] = scaler.fit_transform(reshaped_data)


# In[177]:


test_data = test_data.drop(['F4'],axis = 1)
test_data.shape


# In[179]:


diagnosticplot(test_data, 'pt_F4')


# In[180]:


diagnosticplot(test_data, 'F23')


# In[186]:


#Square Root Transformation
test_data['srt_f23'] = test_data['F23']**(1/2)
diagnosticplot(test_data, 'srt_f23')


# In[187]:


test_data = test_data.drop(['F15'],axis = 1)
test_data.shape


# In[188]:


scaler = StandardScaler()
test_data = scaler.fit_transform(test_data)


# In[189]:


predicted = model.predict(test_data)
predicted


# In[190]:


# Replace the last (empty) column with your prediction
test_df.iloc[:,-1] = predicted


# In[191]:


# Save to the destination file
test_df.to_csv('CE802_P3_Test_Predictions.csv', index=False, float_format='%.8g')


# In[192]:


# IMPORTANT!! Make sure only the last column has changed
assert pd.read_csv('CE802_P3_Test.csv').iloc[:,:-1].equals(pd.read_csv('CE802_P3_Test_Predictions.csv').iloc[:,:-1])


# In[ ]:




