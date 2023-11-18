#!/usr/bin/env python
# coding: utf-8

# ## PART -A

# In[112]:


get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
import seaborn as sns

import pandas as pd
import numpy as np

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline


from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

from sklearn.metrics import confusion_matrix
from sklearn.metrics import plot_confusion_matrix


# ### Loading the Dataset
# 
# **Read the dataset into Pandas DataFrame and storing it in a variable data.**

# In[331]:


data = pd.read_csv("CE802_P2_Data.csv")
data


# ### Analysing the Data

# In[114]:


print(f"The dataset has {data.shape[0]} rows and {data.shape[1]} columns")


# In[115]:


data.describe()


# In[116]:


data.dtypes


# **Checking for any NaN values in the dataset.**

# In[117]:


for col in data.columns:
    print('\t%s: %d' % (col,data[col].isna().sum()))


# **The feature 'F21' has 500 missing values. These null values should be either removed or can be filled with mean, median, mode or with a constatnt value. Discarding the samples with NaN values is losing important information from the dataset. So, here the missing values are filled using the Median of the given field and training the model with 1000 samples.** 

# In[118]:


sns.heatmap(data.isnull(), yticklabels = False, cbar = False, cmap = 'viridis')


# **Check the number of samples for each class and Checking whether the dataset is balanced or imbalanced.**.**

# In[119]:


print("Unique labels : ",data['Class'].unique())
print("Number of samples for each class : ")
print(data['Class'].value_counts())

plt.figure(figsize=(5,5))
sns.countplot(data['Class'])


# **Each class labels have almost equal number of samples. Hence the dataset is a balanced dataset.**

# ## Visualizing the Data using Scatter plot

# In[120]:


sns.pairplot(data, hue = 'Class', diag_kind = 'hist')


# **The scatter plot shows that the data are overlapping and hence they are non-linearly separable.**

# ### Data Cleaning

# In[332]:


data_1 = data.copy()


# **Replacing the NaN values with the median of the given field.**

# In[333]:


print("Before replacing missing values : ")
print(data_1['F21'].head())

imputer = SimpleImputer(missing_values = np.NaN, strategy = 'median')
imputer = imputer.fit(data_1.iloc[:, :21])
data_1.iloc[:, :21] = imputer.transform(data_1.iloc[:, :21])

print("\nAfter replacing missing values by median : ")
print(data_1['F21'].head())


# In[167]:


data_1.isnull().sum()


# **The Dataset is cleaned by replacing the NAN with median values.**

# ### Data Preprocessing

# **Replacing the categorical labels with numerical labels.**
# 
# **The column 'Class' in the dataset has categorical labels. Replacing 'True' with 1 and 'False' with 0 using the LabelEncoder.**

# In[334]:


le = LabelEncoder()
data_1['Class'] = le.fit_transform(data_1['Class'].values)
print("The resulting dataset : ")
data_1.head()


# **Independent  variable 'X' and Dependent variable 'y'**

# In[335]:


X = data_1.drop(['Class'], axis = 'columns')
y = data_1['Class']

print("The shape of 'X' containing the predictors are:", X.shape)
print("The shape of 'y' containing the label are:",y.shape)


# ### Splitting the Data for Training and Validation
# 
# **The dataset is split for training and validation. The training dataset will be used for training and fine tuning the model and the validation dataset will be used for validating the model. The training dataset will be having 80% of the data and the remaining 20% of the data for validation dataset.**

# In[336]:


X_train, X_val, y_train, y_val = train_test_split(X,y, test_size = 0.2, random_state = 10)
print("The shape of X_train : ", X_train.shape)
print("The shape of y_train : ", y_train.shape)
print("The shape of X_val : ", X_val.shape)
print("The shape of y_val : ", y_val.shape)


# ### Models
# 
# **Investigating the performance of a number of Machine learning algorithms for this dataset are,**
# 
# **1.Decision Tree Algorithm**
# 
# **2.Random Forest Algorithm**
# 
# **3.SVM Classifier**

# ### Model Training and Validating

# ### 1.Decision Tree Algorithm

# In[342]:


DT_classifier = DecisionTreeClassifier()
DT_classifier.fit(X_train,y_train)


# In[343]:


y_pred_DT = DT_classifier.predict(X_val) 
print("Training Accuracy for Decision Tree Alg : ", DT_classifier.score(X_train,y_train))
print("Validation Accuracy for Decision Tree Alg : ",DT_classifier.score(X_val,y_val))


# In[344]:


y_pred_DT[:5]


# **Confusion Matrix**

# In[345]:


plot_confusion_matrix(DT_classifier, X_val, y_val)


# ### Visualizing the Decision Tree
# 

# In[346]:


plt.figure(figsize=(20,20))
tree.plot_tree(DT_classifier,filled = True)


# ### Post Pruning Decision Tree with cost complexity pruning
# 
# **DecisionTreeClassifier provides parameters such as min_samples_leaf and max_depth to prevent a tree from overfiting. Cost complexity pruning provides another option to control the size of a tree. This pruning technique is parameterized by the cost complexity parameter, ccp_alpha. Greater values of ccp_alpha increase the number of nodes pruned.**

# In[347]:


pruning = DT_classifier.cost_complexity_pruning_path(X_train, y_train)
ccp_alphas = pruning.ccp_alphas
impurities = pruning.impurities


# In[348]:


clfs = []
for ccp_alpha in ccp_alphas:
    clf = DecisionTreeClassifier(random_state = 0, ccp_alpha= ccp_alpha)
    clf.fit(X_train, y_train)
    clfs.append(clf)
print(f"Number of nodes in the last tree is {clfs[-1].tree_.node_count} with ccp_alpha {ccp_alphas[-1]}")


# ### Setting alpha value for training and validation

# **When ccp_alpha is set to zero and keeping the other default parameters of DecisionTreeClassifier, the tree overfits, leading to a 100% training accuracy and 85% validation accuracy. As alpha increases, more of the tree is pruned, thus creating a decision tree that generalizes better.Setting ccp_alpha=0.01 maximizes the validation accuracy.**

# In[349]:


train_scores = [clf.score(X_train,y_train) for clf in clfs]
val_scores = [clf.score(X_val,y_val) for clf in clfs]

fig, ax = plt.subplots()
ax.set_xlabel("alpha")
ax.set_ylabel("Accuracy")
ax.set_title("Accuracy Vs alpha for Training and Validation data")
ax.plot(ccp_alphas, train_scores, marker = 'o', label = 'train', drawstyle = 'steps-post')
ax.plot(ccp_alphas, val_scores, marker = 'o', label = 'val', drawstyle = 'steps-post')
ax.legend()
plt.show()


# In[351]:


pp_classifier = DecisionTreeClassifier(random_state = 0, ccp_alpha = 0.01)
pp_classifier.fit(X_train,y_train)


# In[352]:


y_pred_pruning = pp_classifier.predict(X_val)
print("Training Accuracy after Post Pruning : ", pp_classifier.score(X_train, y_train))
print("Validation Accuracy after Post Pruning : ", pp_classifier.score(X_val,y_val))


# In[353]:


acc_pp = pp_classifier.score(X_val,y_val)


# **Confusion Matrix for Decision Tree alg after doing Post Pruning**

# In[354]:


plot_confusion_matrix(pp_classifier, X_val, y_val)


# ### Visualizing Decision Tree after Pruning

# In[355]:


plt.figure(figsize=(20,20))
tree.plot_tree(pp_classifier,filled = True)


# ### Random Forest model

# **After experimenting with different training and validation split, training the model with more data improved the performance of the model. Hence splitting the data with the ratio of 90% for training and 10% for validating for Random Forest Classifier.**

# In[294]:


X_train, X_val, y_train, y_val = train_test_split(X,y, test_size = 0.1, random_state = 42)
print("The shape of X_train : ", X_train.shape)
print("The shape of y_train : ", y_train.shape)
print("The shape of X_val : ", X_val.shape)
print("The shape of y_val : ", y_val.shape)


# In[282]:


from sklearn.ensemble import RandomForestClassifier

randomforest_clf = RandomForestClassifier()
randomforest_clf.fit(X_train, y_train)


# In[283]:


y_pred_rf = randomforest_clf.predict(X_val)
print("Training Accuracy for Random Forest Classifier : ",randomforest_clf.score(X_train,y_train))
print("Validation Accuracy for Random Forest Classifier : ",randomforest_clf.score(X_val,y_val))


# ### Performing grid search to find tuned parameters for Random Forest model.

# In[284]:


params = {
    'max_depth': [2,3,5,7,10,20],
    'min_samples_leaf': [5,10,20,50,100,200],
    'n_estimators': [10,25,30,50,100,200],
    'criterion' : ('gini', 'entropy'),
    'max_features' : ('auto','sqrt'),
    'min_samples_split' : (2,4,6)
}

clf = RandomForestClassifier()

grid_search = GridSearchCV(estimator= clf, param_grid= params, scoring= 'accuracy', cv = 10, n_jobs=-1)
grid_search = grid_search.fit(X_train,y_train)


# In[285]:


print("Best score obtained from Grid search : ",grid_search.best_score_)
print("The best parameters to fine tune the Random Forest Classifier : ")
grid_search.best_params_


# In[295]:


rf_classifier = RandomForestClassifier(criterion='gini', max_depth = 20, max_features='auto', min_samples_leaf=5, 
                                  min_samples_split=4,
                                  n_jobs=-1,n_estimators = 200)
rf_classifier.fit(X_train,y_train)


# In[296]:


y_pred_rf = rf_classifier.predict(X_val)
print("Training Accuracy  for Random Forest Alg after fine tuning the Hyperparameter : ",rf_classifier.score(X_train,y_train))
print("Validation Accuracy for Random Forest Alg after fine tuning the Hyperparameter : ",rf_classifier.score(X_val,y_val))


# In[297]:


acc_rf = rf_classifier.score(X_val,y_val)


# **Confusion Matrix for Random Forest Model after fine tuning the Hyperparameter.**

# In[298]:


plot_confusion_matrix(rf_classifier, X_val, y_val)


# ### SVM Classifier

# **As the Decision Tree and the Random Forest Classifier's are insensitive to the scale of the features.** 
# 
# **Standardizing the data using StandardScaler before training and validating the SVM model.** 
# 
# **Here all the values will be transformed in such a way that it will have a standard normal distribution with mean=0 and standard deviation=1..**
# 
# **It is calculated by,**
# 
#                     X_new = (X-mean)/std
# 

# In[217]:


scaler = StandardScaler()
scaler.fit(X)
scaled_data = scaler.transform(X)
print("\nData after Standardization : ")
scaled_data


# In[218]:


#Converting the scaled data to DataFrame
scaled_df = pd.DataFrame(scaled_data, columns = X.columns)
scaled_df.head()


# ### Splitting the scaled data for training and validation

# In[219]:


X_train, X_val, y_train, y_val = train_test_split(scaled_df,y, test_size = 0.1, random_state =42)
print("The shape of X_train : ", X_train.shape)
print("The shape of y_train : ", y_train.shape)
print("The shape of X_val : ", X_val.shape)
print("The shape of y_val : ", y_val.shape)


# ### SVM model training

# In[220]:


from sklearn.svm import SVC

svc_classifier = SVC()
svc_classifier.fit(X_train,y_train)


# In[221]:


y_pred_svc = svc_classifier.predict(X_val)
print("Training Accuracy for SVM model : ",svc_classifier.score(X_train,y_train))
print("Validation Accuracy for SVM model : ",svc_classifier.score(X_val,y_val))


# ### Performing grid search to find tuned parameters for SVM model.

# In[229]:


param_grid = [{"C":[1,10,100,1000],"gamma" : [0.1,0.01,0.001,0.00001],"kernel":["rbf","poly","linear","sigmoid"]}]

grid = GridSearchCV(SVC(),param_grid,refit=True,verbose=3)
grid.fit(X_train,y_train)


# In[223]:


print("Best score obtained from Grid search : ",grid.best_score_)
print("The best parameters to fine tune the SVM model : ")
grid.best_params_


# In[232]:


svm_clf = SVC(C=100, gamma = 0.01, kernel = 'rbf')
svm_clf.fit(X_train,y_train)


# In[233]:


y_pred_svm = svm_clf.predict(X_val)
print("Training Accuracy for SVM after tuning the Hyperparameter : ",svm_clf.score(X_train,y_train))
print("Validation Accuracy for SVM model after tuning Hyperparameter : ",svm_clf.score(X_val,y_val))


# In[234]:


acc_svm = svm_clf.score(X_val,y_val)


# In[236]:


from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV

parameters = {'n_neighbors': [1, 3, 5, 7, 9]}

knn = KNeighborsClassifier()
knn_cv = GridSearchCV(knn, parameters, cv=5)
knn_cv.fit(X_train, y_train)

print('Best hyperparameter setting: {0}.'.format(knn_cv.best_estimator_))
print('Average accuracy across folds of best hyperparameter setting: {0}.'.format(knn_cv.best_score_))
print('Test dataset accuracy of best hyperparameter setting: {0}.'.format(knn_cv.score(X_val, y_val)))


# In[299]:


models = ["Decision Tree", "Random Forest", "SVM"]
scores = []
scores.append(acc_pp)
scores.append(acc_rf)
scores.append(acc_svm)

df = pd.DataFrame()

df['models'] = models
df['scores'] = scores 
df


# In[300]:


sns.set(style='whitegrid')
ax = sns.barplot(y='models', x='scores',data=df)


# **The above barplot shows that the Decision Tree Classifier outperforms with the perfomance of 88% accuracy than the other two models. Hence choosing this model to predict the class labels for the Test dataset.**

# **Building a Pipeline for fitting the model.**

# In[358]:


model_pipeline = Pipeline([('impute_F21',SimpleImputer(missing_values = np.NaN,strategy = 'median')),
                            ('DT_clf',DecisionTreeClassifier(random_state = 0, ccp_alpha = 0.01))])

# fit the pipeline with the training data
model_pipeline.fit(X_train,y_train)

# predict target values on the training data
model_pipeline.predict(X_val)
print("Validation Accuracy : ",model_pipeline.score(X_val,y_val))


# ### Part B

# In[390]:


# HERE YOU WILL USE THIS TEMPLATE TO SAVE THE PREDICTIONS ON THE TEST SET

# Load the test data
test_df = pd.read_csv('CE802_P2_Test.csv')
test_df.head()


# In[391]:


for col in data.columns:
    print('\t%s: %d' % (col,test_df[col].isna().sum()))


# In[392]:


# Make sure you work on a copy
test_data = test_df.iloc[:,:-1].copy()
test_data


# In[393]:


predicted =  model_pipeline.predict(test_data) # CHANGE HERE -- use your previously trained predictor and apply it to test_data
predicted                                              # (test_data can be modified if needed but make sure you don't change the order of the rows)...


# In[394]:


# Replace the last (empty) column with your prediction
test_df.iloc[:,-1] = predicted


# In[395]:


test_df.head()


# In[396]:


# Save to the destination file
test_df.to_csv('CE802_P2_Test_Predictions.csv', index=False, float_format='%.8g')


# In[397]:


# IMPORTANT!! Make sure only the last column has changed
assert pd.read_csv('CE802_P2_Test.csv').iloc[:,:-1].equals(pd.read_csv('CE802_P2_Test_Predictions.csv').iloc[:,:-1])


# In[ ]:





# In[ ]:





# # experimenting with different imputing methods

# In[310]:


data_2 = data.copy()
data_2


# **Replacing the NaN values with Mean**

# In[311]:


print("Before replacing missing values : ")
print(data_2['F21'].head())

imputer = SimpleImputer(missing_values = np.NaN, strategy = 'mean')
imputer = imputer.fit(data_2.iloc[:, :21])
data_2.iloc[:, :21] = imputer.transform(data_2.iloc[:, :21])

print("\nAfter replacing missing values by mean : ")
print(data_2['F21'].head())


# In[313]:


le = LabelEncoder()
data_2['Class'] = le.fit_transform(data_2['Class'].values)
print("The resulting dataset : ")
data_2.head()


# In[314]:


X = data_2.drop(['Class'], axis = 'columns')
y = data_2['Class']

print("The shape of 'X' containing the predictors are:", X.shape)
print("The shape of 'y' containing the label are:",y.shape)


# In[315]:


X_train, X_val, y_train, y_val = train_test_split(X,y, test_size = 0.2, random_state = 10)
print("The shape of X_train : ", X_train.shape)
print("The shape of y_train : ", y_train.shape)
print("The shape of X_val : ", X_val.shape)
print("The shape of y_val : ", y_val.shape)


# In[316]:


DT_classifier = DecisionTreeClassifier()
DT_classifier.fit(X_train,y_train)


# In[317]:


y_pred_DT = DT_classifier.predict(X_val) 
print("Training Accuracy for Decision Tree Alg : ", DT_classifier.score(X_train,y_train))
print("Validation Accuracy for Decision Tree Alg : ",DT_classifier.score(X_val,y_val))


# In[318]:


pruning = DT_classifier.cost_complexity_pruning_path(X_train, y_train)
ccp_alphas = pruning.ccp_alphas
impurities = pruning.impurities


# In[319]:


clfs = []
for ccp_alpha in ccp_alphas:
    clf = DecisionTreeClassifier(random_state = 0, ccp_alpha= ccp_alpha)
    clf.fit(X_train, y_train)
    clfs.append(clf)
print(f"Number of nodes in the last tree is {clfs[-1].tree_.node_count} with ccp_alpha {ccp_alphas[-1]}")


# In[320]:


train_scores = [clf.score(X_train,y_train) for clf in clfs]
val_scores = [clf.score(X_val,y_val) for clf in clfs]

fig, ax = plt.subplots()
ax.set_xlabel("alpha")
ax.set_ylabel("Accuracy")
ax.set_title("Accuracy Vs alpha for Training and Validation data")
ax.plot(ccp_alphas, train_scores, marker = 'o', label = 'train', drawstyle = 'steps-post')
ax.plot(ccp_alphas, val_scores, marker = 'o', label = 'val', drawstyle = 'steps-post')
ax.legend()
plt.show()


# In[329]:


pp_classifier = DecisionTreeClassifier(random_state = 0, ccp_alpha = 0.01)
pp_classifier.fit(X_train,y_train)


# In[330]:


y_pred_pruning = pp_classifier.predict(X_val)
print("Training Accuracy after Post Pruning : ", pp_classifier.score(X_train, y_train))
print("Validation Accuracy after Post Pruning : ", pp_classifier.score(X_val,y_val))


# In[ ]:





# In[ ]:




