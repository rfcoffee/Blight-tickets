#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# %load blight.py
import pandas as pd
import numpy as np
import sys
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve, auc
from sklearn.ensemble import GradientBoostingClassifier

def normalize(data):
    data_mean = np.mean(data, axis=0)
    data_diff = np.max(data, axis=0) - np.min(data, axis=0)
    return (data - data_mean)/data_diff

def learn(trainX1, cvX1, trainy1, cvy1):
    # use logistic regression to learn
#     trainX1, cvX1, trainy1, cvy1 = train_test_split(trainX, trainy, random_state=0)
    trainX1 = normalize(trainX1)
    cvX1 = normalize(cvX1)
    gradboot = GradientBoostingClassifier().fit(trainX1, trainy1)

    # calculate score
    y_score = gradboot.decision_function(cvX1)
    fpr, tpr, _ = roc_curve(cvy1, y_score)
    roc_auc = auc(fpr, tpr)
    return roc_auc
    
def main():    
    try:
        # read train and mca data
        ncol_mca = int((sys.argv[1]))
        train_num = np.load("train_num.npy")
        train_mca = np.load("train_mca_" + sys.argv[1] + ".npy")
        train = np.concatenate((train_num, train_mca), axis=1)
        train_y = np.load("train_y.npy")
        cv_num = np.load("cv_num.npy")
        cv_mca = np.load("cv_mca_" + sys.argv[1] + ".npy")
        cv = np.concatenate((cv_num, cv_mca), axis=1)
        cv_y = np.load("cv_y.npy")
    except:
        print('\033[1;31m\n Error: provide the correct MCA dimension \n\033[1;m')
        return
    
    roc_auc = learn(train, cv, train_y, cv_y)
    print("{:}{:20.10f}".format(ncol_mca, roc_auc))

if __name__ == "__main__":
    main()

