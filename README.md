# Developing a Neural Network Classification Model

## AIM

To develop a neural network classification model for the given dataset.

## Problem Statement

An automobile company has plans to enter new markets with their existing products. After intensive market research, they’ve decided that the behavior of the new market is similar to their existing market.

In their existing market, the sales team has classified all customers into 4 segments (A, B, C, D ). Then, they performed segmented outreach and communication for a different segment of customers. This strategy has work exceptionally well for them. They plan to use the same strategy for the new markets.

You are required to help the manager to predict the right group of the new customers.

## Neural Network Model

Include the neural network model diagram.

## DESIGN STEPS

### STEP 1:
Write your own steps

### STEP 2:

### STEP 3:


## PROGRAM

### Name: 
### Register Number:

```python
import pandas as pd
import numpy as np
import tensorflow as tf
import seaborn as sns
import matplotlib.pylab as plt
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import classification_report,confusion_matrix
df=pd.read_csv("customers.csv")
df.head()
df.info()
df.isnull().sum()
df=df.drop(['ID','Var_1'],axis=1)
df=df.dropna(axis=0)
for i in ['Gender','Ever_Married','Graduated','Profession','Spending_Score','Segmentation']:
    print(i,":",list(df[i].unique()))
Clist=[['Healthcare','Engineer','Lawyer','Artist','Doctor','Homemaker','Entertainment','Marketing',
        'Executive'],['Male', 'Female'],['No', 'Yes'],['No', 'Yes'],['Low', 'Average', 'High']]
enc = OrdinalEncoder(categories=Clist)
df[['Gender','Ever_Married','Graduated','Profession','Spending_Score']]
    =enc.fit_transform(df[['Gender','Ever_Married','Graduated','Profession','Spending_Score']])
le = LabelEncoder()
df['Segmentation'] = le.fit_transform(df['Segmentation'])
scaler=MinMaxScaler()
df[['Age']]=scaler.fit_transform(df[['Age']])
X=df.iloc[:,:-1]
Y=df[['Segmentation']]
ohe=OneHotEncoder()
Y=ohe.fit_transform(Y).toarray()
xtrain,xtest,ytrain,ytest=train_test_split(X,Y,test_size=0.33,random_state=42)
model=Sequential([Dense(6,activation='relu',input_shape=[8]),Dense(10,activation='relu'),
                  Dense(10,activation='relu'),Dense(4,activation='softmax')])
model.compile(optimizer='adam',loss='categorical_crossentropy' ,metrics=['accuracy'])
model.fit(xtrain,ytrain,epochs=2000,batch_size=32,validation_data=(xtest,ytest))
metrics = pd.DataFrame(model.history.history)
metrics[['loss','val_loss']].plot()
ypred = np.argmax(model.predict(xtest), axis=1)
ytrue = np.argmax(ytest,axis=1)
print(confusion_matrix(ytrue,ypred))
print(classification_report(ytrue,ypred))
x_single_prediction = np.argmax(model.predict(X[3:4]), axis=1)
print(x_single_prediction)
print(le.inverse_transform(x_single_prediction))



```

## Dataset Information

Include screenshot of the dataset

## OUTPUT
### Training Loss, Validation Loss Vs Iteration Plot
Include your plot here

### Classification Report

Include Classification Report here

### Confusion Matrix

Include confusion matrix here


### New Sample Data Prediction

Include your sample input and output here

## RESULT
Include your result here
