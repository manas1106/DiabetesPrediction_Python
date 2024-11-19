#!/usr/bin/env python
# coding: utf-8

# Importing the Required Libraries

# In[5]:


import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt


# Data Collection and Analysis
# 
# PIMA Diabetes Dataset

# In[6]:


# loading the diabetes dataset to a pandas DataFrame
diabetes_dataset = pd.read_csv('diabetes.csv') 


# In[7]:


#Printing the first 5 rows of the data set
diabetes_dataset.head()


# This above dataset includes the data for all the females So in that test case Pregnancies column is there.

# In[8]:


#ploting the distribution of outcome
diabetes_dataset.hist(column="Outcome")


# In[5]:


#Number of row and columns in this dataset
diabetes_dataset.shape


# In[6]:


#checking for any null values
diabetes_dataset.isnull().sum()


# In[7]:


#Getting the Statistical Measure of the dataset
diabetes_dataset.describe()


# In[8]:


diabetes_dataset.columns


# In[9]:


diabetes_dataset['Outcome'].value_counts()


# Here 0 represent Non-Diabetic,
#      1 represent Diabetic

# In[10]:


diabetes_dataset.groupby('Outcome').mean()


# In[11]:


# separating the data and labels
X = diabetes_dataset.drop(columns = 'Outcome', axis=1)
Y = diabetes_dataset['Outcome']


# In[12]:


print(X)


# In[13]:


print(Y)


# Data Standardization

# In[14]:


scaler = StandardScaler()


# In[15]:


scaler.fit(X)


# In[16]:


standardized_data = scaler.transform(X)


# In[17]:


print(standardized_data)


# In[18]:


X = standardized_data
Y = diabetes_dataset['Outcome']


# In[19]:


print(X)
print(Y)


# Train Test Split

# In[20]:


X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size = 0.2, stratify=Y, random_state=2)


# In[21]:


print(X.shape, X_train.shape, X_test.shape)


# Training the Model

# In[22]:


classifier = svm.SVC(kernel='linear')


# In[23]:


#training the support vector Machine Classifier
classifier.fit(X_train, Y_train)


# Model Evaluation
# 
# Accuracy Score

# In[24]:


# accuracy score on the training data
X_train_prediction = classifier.predict(X_train)
training_data_accuracy = accuracy_score(X_train_prediction, Y_train)


# In[25]:


print('Accuracy score of the training data : ', training_data_accuracy)


# In[26]:


# accuracy score on the test data
X_test_prediction = classifier.predict(X_test)
test_data_accuracy = accuracy_score(X_test_prediction, Y_test)


# In[27]:


print('Accuracy score of the test data : ', test_data_accuracy)


# Making a Predictive System

# In[28]:


input_data = (5,166,72,19,175,25.8,0.587,51)

# changing the input_data to numpy array
input_data_as_numpy_array = np.asarray(input_data)

# reshape the array as we are predicting for one instance
input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)

# standardize the input data
std_data = scaler.transform(input_data_reshaped)
print(std_data)

prediction = classifier.predict(std_data)
print(prediction)

if (prediction[0] == 0):
  print('The person is not diabetic')
else:
  print('The person is diabetic')


# In[ ]:




