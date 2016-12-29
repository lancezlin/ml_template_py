# libraries needed
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import Imputer
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import StandardScaler
# from StringIO import StringIO
# import global_functions

# Importing data set
datasetA = pd.read_csv('lc_issue_a.csv', skiprows = [0])
#datasetB = pd.read_csv('lc_rej_a.csv', skiprows = [0])
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

# feature scaling transformation
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)
#sc_y = StandardScaler()
#y_train = sc_y.fit_transform(y_train)

# fit models - & DecisionTree 
from sklearn.tree import DecisionTreeClassifier
dt_classifier = DecisionTreeClassifier(criterion = 'gini', random_state = 0) 
dt_classifier.fit(X_train, y_train) # 'gini' is better than 'entropy'

y_pred = dt_classifier.predict(X_test) # predict the test samples

# evaluation of the model - confusion matrix
from sklearn.metrics import confusion_matrix
dt_confusionMatrix = confusion_matrix(y_test, y_pred)
accRate = round(float(float(dt_confusionMatrix[0,0] + dt_confusionMatrix[1,1])/float(len(y_pred)) * 100), 2)
print ("The accuracy of the prediction is %f percent." %accRate)


# fit models - && RandomForest
from sklearn.ensemble import RandomForestClassifier
rf_classifier = RandomForestClassifier(n_estimators = 10, criterion = 'gini', random_state = 0)
rf_classifier.fit(X_train, y_train) # 'gini' is better than 'entropy'

y_pred_rf = rf_classifier.predict(X_test)

# evaluation of the model - confusion matrix
rf_confusionMatrix = confusion_matrix(y_test, y_pred_rf)
rf_accRate = round(float(float(rf_confusionMatrix[0,0] + rf_confusionMatrix[1,1])/float(len(y_pred)) * 100), 2)
print ("The accuracy of the prediction is %f percent." %rf_accRate)













