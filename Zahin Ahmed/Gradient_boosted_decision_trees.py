# -*- coding: utf-8 -*-
"""
Created on Wed Dec 18 21:27:11 2019

@author: Zahin Ahmed
"""
import numpy as np
import pandas as pd


from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import precision_recall_fscore_support
from sklearn.utils.multiclass import unique_labels
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import scale

feature_inp=pd.DataFrame()

print("Reading input file")
feature_inp= pd.read_csv("C:\\Users\\Zahin Ahmed\\Desktop\\CSE465_omics_project\\Data\\curated_input_GYS.csv",index_col=0)
print("input file is imported")



feature_inp=feature_inp.dropna()

X = feature_inp.iloc[:,list(range(1,feature_inp.shape[1]))]
y = feature_inp.iloc[:,0]

X = scale(X)
y = y.as_matrix()

X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.2, random_state=42)

print("Performing gradient boositng")
gbreg = GradientBoostingClassifier(n_estimators=600,max_depth=3,min_samples_leaf=5, min_samples_split=3, subsample=0.8,random_state=10,verbose=True)
gbreg.fit(X_train,y_train)

y_pred=gbreg.predict(X_test)

conf=confusion_matrix(y_test,y_pred , sample_weight=None)

labels = unique_labels(y_test, y_pred)
inp= precision_recall_fscore_support(y_test, y_pred, labels=labels, average=None)
res_conf=conf.ravel().tolist()
res_inp=np.asarray(inp).ravel().tolist()
y_test=np.asfarray(y_test,float)
y_train=np.asfarray(y_train,float)

report=res_conf+res_inp
report=pd.DataFrame(report, index = ['TN','FP','FN','TP','PRC_S','PRC_R','RCL_S','RCL_R','FSc_S','FSc_R','Sc_S','Sc_R'])
report.to_csv("C:\\Users\\Zahin Ahmed\\Desktop\\CSE465_omics_project\\Zahin Ahmed\\report_GB", sep='\t')

print(report)
print("Done!")