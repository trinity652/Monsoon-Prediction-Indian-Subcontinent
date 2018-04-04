
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 12 05:07:57 2018

@author: abhilasha
"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

Spec_dim=9
# Importing the dataset
dataset = pd.read_csv('Aug_9.csv')
X = dataset.iloc[:, 0:Spec_dim+1].values

#Missing Data
from sklearn.preprocessing import Imputer
imputer = Imputer(missing_values = 'NaN', strategy = 'mean', axis = 0)
imputer = imputer.fit(X)
X = imputer.transform(X)

#Feature Scaling
from sklearn import preprocessing 
X=preprocessing.normalize(X)

y=X[:,Spec_dim]
X=X[:,0:Spec_dim]



# Splitting into Training Set and TesT Set
from sklearn.model_selection import train_test_split
X_train, X_Buf, y_train, y_Buf = train_test_split(X, y, test_size = 0.4, random_state = 2)
X_val,X_test,y_val,y_test=train_test_split(X_Buf, y_Buf, test_size = 0.5, random_state = 2)
#print(X_train,'\n',X_val,'\n',X_test,'\n',y_val,'\n',y_test,'\n',y_train)
#Creating the Neural Network
import keras
from keras.models import Sequential
from keras.layers import Dense

#defining the baseline model

regres=Sequential()
regres.add(Dense(output_dim=Spec_dim+1,kernel_initializer='normal',activation='relu',input_dim=Spec_dim))#Adding the input and hidden layer 1
regres.add(Dense(output_dim=1,kernel_initializer='normal'))#Adding the output layer
regres.compile(optimizer='adam',loss='mean_squared_error')#Compiling the ANN

hist =regres.fit(X_train, y_train,batch_size=10,epochs=10000) #Fitting the ANN to the training set

# Predicting the Test set results
y_pred = regres.predict(X_test)
y_pred=np.reshape(y_pred,28)
# Predicting the Test set results
y_pred = regres.predict(X_test)
y_pred=np.reshape(y_pred,27)

y1 = y_pred
y2 = y_test
'''
plt.subplot(2, 1, 1)
plt.plot(y1, 'o-',color='blue')
plt.title('Predicted vs Test Results: ')
plt.ylabel('Predicted Rainfall(Normalized)')

plt.subplot(2, 1, 2)
plt.plot(y2, '.-',color='green')
plt.xlabel('years')
plt.ylabel('Actual Rainfall(Normalized)')

plt.show()
'''
#Line plots together

plt.plot(y1,'.-', color='blue')
plt.plot(y2,'.-', color='green')
plt.xlabel('YEARS')
plt.ylabel('Rainfall Predicted(Blue) vs Actual(Green)')
plt.title('Predicted Vs Test Results')
plt.show()

#RMSE
y3=np.sqrt(sum((y_pred-y_test)**2))
print('The root mean squared error of predicted and real values: ',y3)













