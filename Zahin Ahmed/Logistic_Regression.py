# -*- coding: utf-8 -*-
"""
Created on Thu Dec 19 17:39:35 2019

@author: Zahin Ahmed
"""

import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.utils.multiclass import unique_labels
from sklearn.metrics import confusion_matrix
from sklearn import metrics
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
logreg= LogisticRegression(penalty='l2',solver='lbfgs', max_iter=500, C=1)
logreg.fit(X_train,y_train)

y_pred=logreg.predict(X_test)
conf=confusion_matrix(y_test,y_pred , sample_weight=None)
labels = unique_labels(y_test, y_pred)
res_conf=conf.ravel().tolist()

print (res_conf)

report=pd.DataFrame(res_conf, index = ['TN','FP','FN','TP'])
report.to_csv("C:\\Users\\Zahin Ahmed\\Desktop\\CSE465_omics_project\\Zahin Ahmed\\report_LR", sep='\t')

logreg.predict(X_test)[0:10]
logreg.predict_proba(X_test)[0:10, :]
logreg.predict_proba(X_test)[0:10,1]
y_pred_prob= logreg.predict_proba(X_test)[:,1]
import matplotlib.pyplot as plt

plt.hist(y_pred_prob, bins=0)
plt.xlim(0,1)
plt.title('Histogram of predicted probabilities')
plt.xlabel('predicted probability of resistance')
plt.ylabel('frequency')

from sklearn.preprocessing import binarize
y_pred_class= binarize([y_pred_prob], 0.3)[0]

print(confusion_matrix(y_test, y_pred_class))

fpr, tpr, thresholds= metrics.roc_curve(y_test, y_pred_prob)
plt.plot(fpr,tpr)
plt.xlim([0.0,1.0])
plt.ylim([0.0,1.0])
plt.title('ROC curve for AMX resistance')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.grid(True)

print (metrics.roc_auc_score(y_test,y_pred_prob))
print(logreg)
print(report)
print("Done!")