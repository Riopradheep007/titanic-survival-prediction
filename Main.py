#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 18 11:21:28 2020

@author:pradheep
"""

from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
import pandas as pd


df=pd.read_csv("formulated_train.csv")
print(df.head())

x=df.drop(['Survived'],axis=1)

y=df['Survived']


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
featureScores.columns = ['columns','Score']  #naming the dataframe columns

#print(featureScores)

print(featureScores.nlargest(10,'Score'))

#################################################################################3
#withoutout(hyperparameter tunning)

from sklearn.model_selection import train_test_split 
from sklearn import tree

# Decision Tree Classifier

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)

                                           
model = tree.DecisionTreeClassifier()
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





#########################################################################################3
'''hyperparameter tunning our model'''

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



################################################################################################3
#with hyper parameter tunning

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
#print(len(predictions))

from sklearn.metrics import confusion_matrix

accuracy=confusion_matrix(y_test,predictions)

print("confusion_matrix:",accuracy)

from sklearn.metrics import accuracy_score

accuracy=accuracy_score(y_test,predictions)
print("accuracy_score:",accuracy)

####################################################################
#clearning the test data
test=pd.read_csv("formulated_test.csv")


predictions = model.predict(test)

#print(len(predictions))
#################################################################



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
