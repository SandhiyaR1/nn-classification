# Developing a Neural Network Classification Model

## AIM
To develop a neural network classification model for the given dataset.
## Problem Statement

An automobile company has plans to enter new markets with their existing products. After intensive market research, they’ve decided that the behavior of the new market is similar to their existing market.

In their existing market, the sales team has classified all customers into 4 segments (A, B, C, D ). Then, they performed segmented outreach and communication for a different segment of customers. This strategy has work exceptionally well for them. They plan to use the same strategy for the new markets.

You are required to help the manager to predict the right group of the new customers.

## Neural Network Model

![image](https://github.com/user-attachments/assets/a6ff23d7-df37-4320-bfc5-4618ccbcbd99)

## DESIGN STEPS

### STEP 1:
Prepare and preprocess the customer data by cleaning, encoding categorical features, and splitting it into training and testing sets.
### STEP 2:
Build and train a neural network using TensorFlow/Keras to predict customer segments based on the preprocessed data.

### STEP 3:
Evaluate the model’s performance, save the trained model, and use it to predict customer segments for new data.



## PROGRAM

### Name: SANDHIYA R
### Register Number: 212222230129

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
categories_list=[['Male', 'Female'],
           ['No', 'Yes'],
           ['No', 'Yes'],
           ['Healthcare', 'Engineer', 'Lawyer', 'Artist', 'Doctor',
            'Homemaker', 'Entertainment', 'Marketing', 'Executive'],
           ['Low', 'Average', 'High']
           ]
customers_1 = customer_df_cleaned.copy()

customers_1[['Gender',
             'Ever_Married',
              'Graduated','Profession',
              'Spending_Score']] = enc.fit_transform(customers_1[['Gender',
                                                                 'Ever_Married',
                                                                 'Graduated','Profession',
                                                                 'Spending_Score']])
le = LabelEncoder()
customers_1['Segmentation'] = le.fit_transform(customers_1['Segmentation'])
customers_1.dtypes
customers_1 = customers_1.drop('ID',axis=1)
customers_1 = customers_1.drop('Var_1',axis=1)
# Calculate the correlation matrix
corr = customers_1.corr()
# Plot the heatmap
sns.heatmap(corr,
        xticklabels=corr.columns,
        yticklabels=corr.columns,
        cmap="BuPu",
        annot= True)
sns.pairplot(customers_1)
sns.distplot(customers_1['Age'])
plt.figure(figsize=(10,6))
sns.countplot(customers_1['Family_Size'])
customers_1['Segmentation'].unique()
X=customers_1[['Gender','Ever_Married','Age','Graduated','Profession','Work_Experience','Spending_Score','Family_Size']].values
y1 = customers_1[['Segmentation']].values
one_hot_enc = OneHotEncoder()
one_hot_enc.fit(y1)
y = one_hot_enc.transform(y1).toarray()
xtrain,xtest,ytrain,ytest=train_test_split(X,Y,test_size=0.33,random_state=42)
scaler_age=MinMaxScaler()
scaler_age.fit(X_train[:,2].reshape(-1,1))
X_train_scaled[:,2] = scaler_age.transform(X_train[:,2].reshape(-1,1)).reshape(-1)
X_test_scaled[:,2] = scaler_age.transform(X_test[:,2].reshape(-1,1)).reshape(-1)
model=Sequential([Dense(6,activation='relu',input_shape=[8]),Dense(10,activation='relu'),
                  Dense(10,activation='relu'),Dense(4,activation='softmax')])
model.compile(optimizer='adam',loss='categorical_crossentropy' ,metrics=['accuracy'])
model.fit(xtrain,ytrain,epochs=2000,batch_size=32,validation_data=(xtest,ytest))
metrics = pd.DataFrame(model.history.history)
metrics[['loss','val_loss']].plot()
ypred = np.argmax(model.predict(xtest), axis=1)
x_test_predictions = np.argmax(model.predict(X_test_scaled), axis=1)
x_test_predictions.shape
y_test_truevalue = np.argmax(y_test,axis=1)
y_test_truevalue.shape
print(confusion_matrix(y_test_truevalue,x_test_predictions))
print(classification_report(y_test_truevalue,x_test_predictions))
x_single_prediction = np.argmax(ai_brain.predict(X_test_scaled[1:2,:]), axis=1)
print(x_single_prediction)
print(le.inverse_transform(x_single_prediction))
```
## OUTPUT

## Dataset Information
![image](https://github.com/user-attachments/assets/cd11cae9-96b8-420b-bdf6-d9d79ee2b4a1)

### Training Loss, Validation Loss Vs Iteration Plot
![image](https://github.com/user-attachments/assets/5b37440f-9a23-4e50-b490-49ce015935d2)

### Classification Report

![image](https://github.com/user-attachments/assets/886863e7-e292-472c-a19f-e3ec9d5a06f1)

### Confusion Matrix

![image](https://github.com/user-attachments/assets/2a721597-a9a2-43f2-a5ca-172e614abacd)


### New Sample Data Prediction

![image](https://github.com/user-attachments/assets/d0fabf1d-e4f7-4d67-acbd-1952ede18a8d)

## RESULT
A neural network classification model for the given dataset is successfully developed.
