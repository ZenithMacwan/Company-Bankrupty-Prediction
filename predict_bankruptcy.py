#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')


# In[3]:


df = pd.read_csv('./data.csv')
df.head()


# In[6]:


df.shape
#high dimension data


# In[8]:


df.isnull().sum()


# In[9]:


x = df.iloc[:,1:]
y = df.iloc[:, :1]


# In[10]:


# importing required libraries

import matplotlib.pyplot as plt
import seaborn as sns
# plot heatmap to check the correlation coefficients

plt.figure(figsize = (16, 10))
sns.heatmap(df.corr(), annot = True, cmap="YlGnBu")
plt.show()


# In[12]:


def correlation(dataset,threshold):
    col_corr = set()
    corr_matrix = dataset.corr()
    for i in range(len(corr_matrix.columns)):
        for j in range(i):
            if abs(corr_matrix.iloc[i,j]) > threshold:
                colname = corr_matrix.columns[i]
                col_corr.add(colname)
    return col_corr
cor_var = correlation(df,0.9)
cor_var


# # Train test split

# In[13]:


from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
xtrain, xtest, ytrain, ytest = train_test_split(x,y,test_size = 0.25, random_state = 102)


# In[22]:



from imblearn.over_sampling import SMOTE
class_distribution = y.value_counts()
print("Class Distribution:")
print(class_distribution)


# In[24]:


# Perform SMOTE oversampling on the training data
smote = SMOTE(random_state=42)
x, y = smote.fit_resample(x, y)


# In[26]:


from imblearn.over_sampling import SMOTE
class_distribution = y.value_counts()
print("Class Distribution:")
print(class_distribution)


# In[29]:


sc = StandardScaler()
xtrain = sc.fit_transform(xtrain)
xtest = sc.fit_transform(xtest)
xtrain, xtest, ytrain, ytest = train_test_split(x,y,test_size = 0.25, random_state = 102)


# In[30]:



# Create a Random Forest classifier
rf_classifier = RandomForestClassifier()

# Train the classifier on the training data
rf_classifier.fit(xtrain, ytrain)

# Make predictions on the test data
y_pred = rf_classifier.predict(xtest)

# Evaluate the model's performance
accuracy = accuracy_score(ytest, y_pred)
print(f"Accuracy: {accuracy:.2f}")

# Get a detailed classification report
print(classification_report(ytest, y_pred))


# In[34]:


from sklearn.linear_model import LogisticRegression
lr = LogisticRegression()
lr.fit(xtrain,ytrain)
y_pred = lr.predict(xtest)
# Evaluate the model's performance
accuracy = accuracy_score(ytest, y_pred)
print(f"Accuracy: {accuracy:.2f}")

# Get a detailed classification report
print(classification_report(ytest, y_pred))


# # Feature Engineering
# 

# In[79]:


from sklearn.decomposition import PCA
n_components = 22
# Choose the number of principal components
pca = PCA(n_components=n_components)
xtrain2 = pca.fit_transform(xtrain)
xtest2 = pca.transform(xtest)


# In[80]:


lr1 = LogisticRegression()
lr1.fit(xtrain2,ytrain)
y_pred = lr1.predict(xtest2)
# Evaluate the model's performance
accuracy = accuracy_score(ytest, y_pred)
print(f"Accuracy: {accuracy:.2f}")

# Get a detailed classification report
print(classification_report(ytest, y_pred))


# In[82]:


from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV

param = {'penalty': ['l1', 'l2'],
    'C': [0.1, 1, 10],
    'solver': ['liblinear', 'saga'],
    'max_iter': [100, 200, 500],
    'class_weight': [None, 'balanced']}

random_search = GridSearchCV(lr1, param, cv=5)
random_search.fit(xtrain2, ytrain)
print("best hyperparameters: ", random_search.best_params_)

best_model = random_search.best_estimator_

# Make predictions on the test set using the best model
y_pred = best_model.predict(xtest2)

# Evaluate the model
report = classification_report(ytest, y_pred)
print(report)


# In[85]:


print("best hyperparameters: "," 'C': 0.1, 'class_weight': 'balanced', 'max_iter': 100, 'penalty': 'l2', 'solver': 'liblinear'")


# In[86]:


from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier


# In[87]:


# Create and train the classification models
models = {
    'Logistic Regression': LogisticRegression(random_state=42),
    'Decision Tree': DecisionTreeClassifier(random_state=42),
    'Random Forest': RandomForestClassifier(random_state=42),
    'SVM': SVC(random_state=42),
    'K-Nearest Neighbors': KNeighborsClassifier()
}

for name, model in models.items():
    print(f"Training {name}...")
    model.fit(xtrain2, ytrain)
    
    # Make predictions on the test data
    y_pred = model.predict(xtest2)
    
    # Evaluate the model's performance
    accuracy = accuracy_score(ytest, y_pred)
    print(f"Accuracy: {accuracy:.2f}")
    
    # Get a detailed classification report
    print(f"Classification Report for {name}:")
    print(classification_report(ytest, y_pred))
    print('-' * 40)


# In[90]:


base_models = [
    LogisticRegression(random_state=42),RandomForestClassifier(n_estimators=100, random_state=42),
    DecisionTreeClassifier(random_state=42),SVC(random_state=42),KNeighborsClassifier()
]

predictions = np.zeros((xtest.shape[0], len(base_models)))
meta_model = LogisticRegression()

# Create an array to hold the predictions of base models on the test set
predictions = np.zeros((xtest.shape[0], len(base_models)))

# Train the base models and make predictions on the test set
for i, model in enumerate(base_models):
    model.fit(xtrain, ytrain)
    y_pred = model.predict(xtest)
    predictions[:, i] = y_pred

# Train the meta-model on the predictions of base models
meta_model.fit(predictions, ytest)

# Make final predictions using the stacking model
final_predictions = meta_model.predict(predictions)

# Evaluate the stacking model
accuracy = accuracy_score(ytest, final_predictions)
print("Stacking Model Accuracy:", accuracy)


# In[ ]:




