# Logistic_Regression_Purchase_Item_Predict_Future_Data
LogisticRegression classifier used for predicting the item is Purchase or not along with Predict Future_Data
Logistic_Regression_Purchase_Item_Predict_Future_Data

LogisticRegression classifier used for predicting the item is Purchase or not along with Predict Future_Data
Logistic Regression
Importing the libraries

import numpy as np import matplotlib.pyplot as plt import pandas as pd
Importing the dataset

dataset=pd.read_csv(r"C:\Users\SAI\Desktop\Data science\June month\1st 15. Logistic regression with future prediction\15. Logistic regression with future prediction\Social_Network_Ads.csv") #this datasset contian information of user and socianl network, those features are - userid,gender,age,salary,purchased #social network has several business client which can put their into social networks and one of the client is car company , this company has newly lunched XUV in rediculous price or high price #we will see which of the user in this social network are going to buy brand new xuv car #Last column tell us user purchased the car yes-1 // no-0 & we are going to build the model that is goint to predict if the user is going to buy xuv or not based on 2 variable based on age & estimated salery #so our matrix of feature is only these 2 column & we gonna find some corelation b/w age and estimated salary of user and his decission to purchase the car [yes or no] #so i need 2 index and rest of index i will remove for this i have to use slicing operator #1 means - the user going to buy the car & 0 means - user is not going to buy the car

X = dataset.iloc[:, [2, 3]].values y = dataset.iloc[:, -1].values
Splitting the dataset into the Training set and Test set

from sklearn.model_selection import train_test_split #for this observation let me selcted as 100 observaion for test set

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 0)

#we are going to predict which users are going to predit xuv,
Feature Scaling

from sklearn.preprocessing import StandardScaler sc = StandardScaler() X_train = sc.fit_transform(X_train) X_test = sc.transform(X_test) #we mentioned feature scaling only to independent variable not dependent variable at all

#datapreprocessing done guys upto this part

#******************************************************************************************

#Next step is we are going to build the logistic model and appy this model into our dataset #This is linear model library thats why we called from sklear.linear_model
Training the Logistic Regression model on the Training set

from sklearn.linear_model import LogisticRegression classifier = LogisticRegression(penalty='l2', solver='sag') classifier.fit(X_train, y_train) #we have to fit the logistic regression model to our training set
Predicting the Test set results

y_pred = classifier.predict(X_test) #now you compare X_test with y_pred, x-test we have age and salary , #if u look at the first observation this user is not be able to buy the car but if you look at observation 7 then that user is going to buy the car #in this case logistic regression model classify the which users are going to buy the car or not

#we build our logistic model and fit it to the training set & we predict our test set result

#now we will use the confusion matrix to evalute
Making the Confusion Matrix

from sklearn.metrics import confusion_matrix cm = confusion_matrix(y_test, y_pred) print(cm) #we can say that 65 + 24 = 89 correct prediction we found & 8+3 = 11 incorrect prediction made
This is to get the Models Accuracy

from sklearn.metrics import accuracy_score ac = accuracy_score(y_test, y_pred) print(ac)

bias = classifier.score(X_train, y_train) print(bias)

variance = classifier.score(X_test, y_test) print(variance)
This is to get the Classification Report

from sklearn.metrics import classification_report cr = classification_report(y_test, y_pred) cr

#****************** 10. check the future data with predicated data furture_dataset=pd.read_csv(r'C:\Users\ankus\OneDrive\Desktop\April\28th,29th_April\TASK-25_Purchase\future prediction _ 2.csv')

X_new=furture_dataset.iloc[:,[2,3]].values X_new=sc.fit_transform(X_new) y_Logi_pred=classifier.predict(X_new) #********************* 11. Dataframe Creation import os df1=pd.DataFrame(y_Logi_pred) df1=df1.rename(columns={'NAN':'Index', 0:'modelpredicated'}) 
