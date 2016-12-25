# libraries needed
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from StringIO import StringIO

# Importing data set
datasetA = pd.read_csv('lc_issue_a.csv', skiprows = [0])
datasetB = pd.read_csv('lc_rej_a.csv', skiprows = [0])
dataset = pd.concat([datasetA, datasetB], axis = 0)


X = dataset.iloc[:, 1:2].values
y = dataset.iloc[:, 2].values

#