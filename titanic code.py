#!/usr/bin/env python
# coding: utf-8

# In[159]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, train_test_split, RandomizedSearchCV
from sklearn.metrics import classification_report
from sklearn.linear_model import SGDClassifier
from sklearn.svm import SVC


# In[57]:


data = pd.read_csv('train.csv')


# # exploring the data

# In[25]:


data


# In[17]:


data.info()


# In[22]:


data.isna().sum()


# In[23]:


data.describe()


# In[27]:


data.columns


# In[28]:


data.head()


# In[29]:


data.shape


# ## The data has 891 rows and 12 columns
# ### Problems with the data: 
# * Name, Sex, Ticket, Cabin, and Embarked should be changed to categories because currently they are objects
# * Age, cabin, and Embarked have missing values

# In[30]:


#backup
data2 = data


# In[58]:


# changeing objects to categories
for label, content in data.items():
    if pd.api.types.is_string_dtype(content) == True:
        data[label] = content.astype('category')


# In[59]:


data.info()


# In[35]:


data.head()


# In[60]:


#filling out missing values for age
data['Age'].fillna(data['Age'].median(), inplace=True)


# In[61]:


data.isna().sum()


# In[62]:


#filling out missing values for cabin and embarked
for label, content in data.items():
    if not pd.api.types.is_numeric_dtype(content) == True:
        data[label] = pd.Categorical(content).codes + 1


# In[63]:


data


# In[64]:


data


# In[76]:


#splitting the data and making the model
np.random.seed(0)
x = data.drop('Survived', axis=1)
y = data['Survived']

x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.2)
model = RandomForestClassifier()

model.fit(x_train,y_train)
model.score(x_test, y_test)


# In[95]:


cross_val_score(model, x,y, cv=20).mean()


# In[96]:


y_preds = model.predict(x_test)


# In[98]:


print(classification_report(y_test, y_preds))


# In[100]:


#improving the model
grid={'n_estimators': [10,100,200,500,1000,1200],
      'max_depth':[5, 10, 20, 30, 40],
      'max_features':[0.2,'auto', 'sqrt'],
      'min_samples_split':[2,4,6],
      'min_samples_leaf':[1,2,4]}
np.random.seed(0)
x = data.drop('Survived', axis=1)
y = data['Survived']

x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.2)
model = RandomForestClassifier(n_jobs=-1)
a_model = RandomizedSearchCV(estimator=model,
                            param_distributions=grid,
                            n_iter=40, 
                            cv=5,
                            verbose=True)
a_model.fit(x_train,y_train)
a_model.score(x_test, y_test)


# In[103]:


y_preds2 = a_model.predict(x_test)
print(classification_report(y_test, y_preds2))


# In[157]:


# testing other models
def perfect_model(model):
    np.random.seed(0)
    x = data.drop('Survived', axis=1)
    y = data['Survived']
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)
    clf = model()
    clf.fit(x_train, y_train)
    score = clf.score(x_test,y_test)
    return score


# In[161]:


perfect_model(SVC)


# In[164]:


perfect_model(SGDClassifier)


# In[162]:


test = pd.read_csv('test.csv')


# In[78]:


test


# In[80]:


test.isna().sum()


# # fixing the test data

# In[82]:


test['Age'].fillna(test['Age'].median(),inplace=True)


# In[83]:


test.info()


# In[84]:


for label, content in test.items():
    if pd.api.types.is_string_dtype(content) == True:
        test[label] = content.astype('category')


# In[85]:


for label, content in test.items():
    if not pd.api.types.is_numeric_dtype(content) == True:
        test[label] = pd.Categorical(content).codes + 1


# In[86]:


test.info()


# In[89]:


test['Fare'].fillna(test['Fare'].mean(), inplace=True)


# In[91]:


test.isna().sum()


# In[108]:


test


# In[104]:


test_preds = a_model.predict(test)


# In[137]:


sub = [test['PassengerId'],test_preds]
submission = pd.DataFrame(sub).T


# In[138]:


submission


# In[149]:


submission.rename({'Passenger':'PassengerId','Unnamed 0': 'Survived'}, axis=1, inplace=True)


# In[150]:


submission


# In[151]:


submission.to_csv('submission.csv', index=False)


# In[ ]:




