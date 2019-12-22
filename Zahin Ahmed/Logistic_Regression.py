# -*- coding: utf-8 -*-
"""
Created on Thu Dec 19 17:39:35 2019

@author: Zahin Ahmed
"""

import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_recall_fscore_support
from sklearn.utils.multiclass import unique_labels
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import scale

feature_inp=pd.DataFrame()

print("Reading input file")
feature_inp= pd.read_csv("C:\\Users\\Zahin Ahmed\\Desktop\\CSE465_omics_project\\Data\\curated_input_GYS.csv",index_col=0)
print("input file is imported")

print("No. of features:")
range(1,feature_inp.shape[1])

feature_inp=feature_inp.dropna()
 

X = feature_inp.iloc[:,list(range(1,feature_inp.shape[1]))]
y = feature_inp.iloc[:,0]

X = scale(X)
y = y.as_matrix()

X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.2, random_state=42)
print("Performing Logistic Regression")
logreg= LogisticRegression(penalty='l2',solver='lbfgs', max_iter=500, C=4)
logreg.fit(X_train,y_train)

y_pred=logreg.predict(X_test)
conf=confusion_matrix(y_test,y_pred , sample_weight=None)
labels = unique_labels(y_test, y_pred)
res_conf=conf.ravel().tolist()



report=pd.DataFrame(res_conf, index = ['TN','FP','FN','TP'])
report.to_csv("C:\\Users\\Zahin Ahmed\\Desktop\\CSE465_omics_project\\Zahin Ahmed\\report_LR", sep='\t')

print(logreg)
print(report)
print("Done!")