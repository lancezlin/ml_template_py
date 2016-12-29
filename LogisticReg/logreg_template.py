# -*- coding: utf-8 -*-
"""
Created on Wed Dec 28 23:02:47 2016

@author: lancel
"""

# libraries needed
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import Imputer
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
# from StringIO import StringIO
# import global_functions

# Importing data set
datasetA = pd.read_csv('../datasets/lc_issue_a.csv', skiprows = [0])
#datasetB = pd.read_csv('../datasets/lc_rej_a.csv', skiprows = [0])
#dataset = pd.concat([datasetA, datasetB], axis = 0)
columnsToRemain = ['loan_amnt', 'term', 'emp_length',
                   'annual_inc', 'int_rate', 
                   'grade', 'total_acc', 'loan_status']
dataset = datasetA[columnsToRemain]

# Cleaning data
mapStatus = {'Fully Paid' : 1, 'Current' : 1, 'Charged Off' : 0, 
             'Default' : 1, 'In Grace Period' : 1, 
             'Late (31-120 days)' : 0, 'Late (16-30 days)' : 0, 
             'Does not meet the credit policy. Status:Fully Paid' : 1, 
             'Does not meet the credit policy. Status:Charged Off' : 0}
datasetC = dataset.dropna(how = 'all')
datasetC['term'] = datasetC['term'].str.extract('(\d\d)', expand=True)
datasetC['emp_length'] = datasetC['emp_length'].str.extract('([0-9]+)', expand=True).astype('float')
datasetC['int_rate'] = datasetC['int_rate'].replace('%', '', regex=True).astype('float')/100
datasetC['health'] = datasetC['loan_status'].map(mapStatus)

X = datasetC.iloc[:, 0:-2].values
y = datasetC.iloc[:, -1].values

imputer = Imputer(missing_values = 'NaN', strategy = 'median', axis = 0)
imputer = imputer.fit(X[:, [2, 3, 6]])
X[:, [2, 3, 6]] = imputer.transform(X[:, [2, 3, 6]])


# Encode categorical variables in X and y
labelencoder_X = LabelEncoder()
X[:, 5] = labelencoder_X.fit_transform(X[:, 5])
onehotencoder_X = OneHotEncoder(categorical_features = [5])
X = onehotencoder_X.fit_transform(X).toarray()

labelencoder_y = LabelEncoder()
y = labelencoder_y.fit_transform(y)

# split into train & test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)

# feature scaling transformation - two methods (minmax vs stardard)
sc_X = StandardScaler()
X_train_ss = sc_X.fit_transform(X_train)
X_test_ss = sc_X.transform(X_test)
#sc_y = StandardScaler()
#y_train = sc_y.fit_transform(y_train)

mm_X = MinMaxScaler()
X_train_mm = mm_X.fit_transform(X_train)
X_test_mm = mm_X.transform(X_test)

# fit models - Logistic Regression Models
from sklearn.linear_model import LogisticRegression

ss_lg_classifier = LogisticRegression(random_state = 0)
mm_lg_classifier = LogisticRegression(random_state = 0)

ss_lg_classifier.fit(X_train_ss, y_train)
mm_lg_classifier.fit(X_train_mm, y_train)

# predict the test results
ss_y_pred = ss_lg_classifier.predict(X_test)
mm_y_pred = mm_lg_classifier.predict(X_test)

# evaluation of the models
from sklearn.metrics import confusion_matrix

cm_ss = confusion_matrix(ss_y_pred, y_test)
cm_mm = confusion_matrix(mm_y_pred, y_test)

ss_accrate = round(float(float(cm_ss[0,0] + cm_ss[1,1])/float(len(y_pred)) * 100), 2)
mm_accrate = round(float(float(cm_mm[0,0] + cm_mm[1,1])/float(len(y_pred)) * 100), 2)

print ("The prediction accuracy is %f percent." %ss_accrate)
print ("The prediction accuracy is %f percent." %mm_accrate)

