#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 18 11:21:28 2020

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
'''
missing_data=missingno.matrix(x,figsize = (10,10))
print(missing_data)'''

'''
sns.set_style('whitegrid')
sns.countplot(x='Survived',hue='Pclass',data=x,palette='rainbow')'''

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
'''
x= pd.concat([x,y],axis=1)

#train_df=pd.concat([df['SalePrice'],train_df],axis=1)
cs=x
cs.to_csv('../formulated_train.csv', index=False)
print('Submission CSV is ready!')'''
##############################################################################333
#feature importance in this model

from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
#apply SelectKBest class to extract top 10 best features
bestfeatures = SelectKBest(score_func=chi2, k=10)
fit = bestfeatures.fit(x,y)

dfscores = pd.DataFrame(fit.scores_)
dfcolumns = pd.DataFrame(x.columns)

#concat two dataframes for better visualization 
featureScores = pd.concat([dfcolumns,dfscores],axis=1)
featureScores.columns = ['Specs','Score']  #naming the dataframe columns

#print(featureScores)

print(featureScores.nlargest(10,'Score'))

#################################################################################3


from sklearn.model_selection import train_test_split 
from sklearn import tree

# Decision Tree Classifier

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)

                                           
model = tree.DecisionTreeClassifier(criterion= 'gini',
                                             max_depth= 3,
                                             max_features= 9,
                                             min_samples_leaf= 1,
                                             splitter='best')
model.fit(x_train,y_train)
 # Cross Validation 

predictions = model.predict(x_test)
#print("preddicted value",pcriterion='gini'redictions)

from sklearn.metrics import confusion_matrix

accuracy=confusion_matrix(y_test,predictions)

print("confusion_matrix:",accuracy)

from sklearn.metrics import accuracy_score

accuracy=accuracy_score(y_test,predictions)
print("accuracy_score:",accuracy)

#print(len(predictions))
'''
from sklearn.model_selection import RandomizedSearchCV

from scipy.stats import randint

est=tree.DecisionTreeClassifier()
rf_p_dist={'max_depth':[3,5,10,None],
           
           'max_features':randint(1,10),
           'criterion':['entropy','gini'],
          
           'min_samples_leaf':randint(1,4),
           
        
    
        }
def hypertunning_rscv(est,p_distr,nbr_iter,x,y):
    rdmsearch=RandomizedSearchCV(est,param_distributions=p_distr,
                                 n_jobs=-1,n_iter=nbr_iter,cv=9)
    rdmsearch.fit(x,y)
    ht_params=rdmsearch.best_params_
    ht_score=rdmsearch.best_score_
    return ht_params,ht_score


rf_parameters,rf_ht_score=hypertunning_rscv(est,rf_p_dist,40,x,y)

'''
#cleaning tst dataset
################################################################################################3
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
print('Submission CSV is ready!')


predictions = model.predict(test)

#print(len(predictions))





# Create a submisison dataframe and append the relevant columns
submission = pd.DataFrame()
submission['PassengerId'] = test['PassengerId']
submission['Survived'] = predictions # our model predictions on the test dataset
submission.head()


# Let's convert our submission dataframe 'Survived' column to ints
submission['Survived'] = submission['Survived'].astype(int)
print('Converted Survived column to integers.')

print(submission.head())




# Are our test and submission dataframes the same length?
if len(submission) == len(test):
    print("Submission dataframe is the same length as test ({} rows).".format(len(submission)))
else:
    print("Dataframes mismatched, won't be able to submit to Kaggle.")

# Convert submisison dataframe to csv for submission to csv 
# for Kaggle submisison
submission.to_csv('../decisiontree_submission3.csv', index=False)
print('Submission CSV is ready!')
