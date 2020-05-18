#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 18 14:51:02 2020

@author: kpr
"""

import missingno
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib as plt
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, label_binarize
# Start Python Imports
#import math, time, random, datetime


df=pd.read_csv("/home/kpr/Music/machine learning/daat analysis/your-first-kaggle-submission-master/data/train.csv")
print(df.head())

x=df.drop(['Survived','Cabin','Name','Ticket'],axis=1)

y=df['Survived']

missing_data=missingno.matrix(x,figsize = (10,10))
print(missing_data)


sns.set_style('whitegrid')
sns.countplot(x='Survived',hue='Pclass',data=x,palette='rainbow')

x['Sex'] = np.where(x['Sex'] == 'female', 1, 0)

sns.boxplot(x='Pclass',y='Age',data=x)

def impute_age(cols):
    Age = cols[0]
    Pclass = cols[1]
    
    if pd.isnull(Age):

        if Pclass == 1:
            return 37

        elif Pclass == 2:
            return 29

        else:
            return 24

    else:
        return Age
    
x['Age'] =x[['Age','Pclass']].apply(impute_age,axis=1)

#print(x.Age.isnull().sum())

df_sex_one_hot = pd.get_dummies(x['Embarked'], 
                                prefix='Embarked')


# Combine the one hot encoded columns with df_con_enc
x= pd.concat([x,df_sex_one_hot],axis=1)
                       

# Drop the original categorical columns (because now they've been one hot encoded)
x=x.drop(['Embarked'], axis=1)

print(len(x))
print(x.head())

x= pd.concat([x,y],axis=1)

#train_df=pd.concat([df['SalePrice'],train_df],axis=1)
cs=x
cs.to_csv('../formulated_train.csv', index=False)
print('modified train CSV is ready!')





###############################################################################3

#clearning the test data
test=pd.read_csv('data/test.csv')
#print('the test len:',len(test))

#print(test.head())

# One hot encode the columns in the test data frame (like X_train)
test_embarked_one_hot = pd.get_dummies(test['Embarked'], 
                                       prefix='Embarked')

# Combine the test one hot encoded columns with test
test = pd.concat([test, 
                  test_embarked_one_hot], axis=1)

test=test.drop(['Embarked','Name','Cabin','Ticket'], axis=1)

test['Sex'] = np.where(test['Sex'] == 'female', 1, 0)


def impute_age1(cols):
    Age = cols[0]
    Pclass = cols[1]
    
    if pd.isnull(Age):

        if Pclass == 1:
            return 37

        elif Pclass == 2:
            return 29

        else:
            return 24

    else:
        return Age
    
test['Age'] =test[['Age','Pclass']].apply(impute_age1,axis=1)




print(test.isnull().sum())

#train_df=pd.concat([df['SalePrice'],train_df],axis=1)
csv=test
csv.to_csv('../formulated_test.csv', index=False)
print('modified test CSV is ready!')


