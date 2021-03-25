import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn import tree
from sklearn import ensemble
from sklearn.svm import SVC
from joblib import dump, load
from utils.helpers import *
from utils.iFeature_helper import *
from sklearn.utils import resample
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import GridSearchCV

with open('data/train_data_bert.npy', 'rb') as f:
    data_bert = np.load(f, allow_pickle=True)
with open('data/train_labels_bert.npy', 'rb') as f:
    labels_bert = np.load(f, allow_pickle=True)

with open('data/independent_2_data_bert.npy', 'rb') as f:
    data_bert_val = np.load(f, allow_pickle=True)
with open('data/independent_2_labels_bert.npy', 'rb') as f:
    labels_bert_val = np.load(f, allow_pickle=True)

with open('data/independent_1_data_bert.npy', 'rb') as f:
    data_bert_val_1 = np.load(f, allow_pickle=True)
with open('data/independent_1_labels_bert.npy', 'rb') as f:
    labels_bert_val_1 = np.load(f, allow_pickle=True)

# data_bert = data_bert[:, 0]
data_bert_val = data_bert_val[:, 0]
data_bert_val_1 = data_bert_val_1[:, 0]
#
# # resampling
# posi = data_bert[labels_bert == 1]
# nega = data_bert[labels_bert == 0]
# # up resample
# posi = resample(posi, replace=True, n_samples=int(len(nega)), random_state=7)
# # down resample
# # nega = resample(nega, replace=True, n_samples=int(len(posi)), random_state=1010)
# # concat data
# print(len(posi))
# print(len(nega))
# labels_posi = [1 for i in range(len(posi))]
# labels_nega = [0 for i in range(len(nega))]
# data_bert = np.append(posi, nega, axis=0)
# labels_bert = np.append(labels_posi, labels_nega)
# data_bert, labels_bert = balance_data(data_bert, labels_bert)

# SMOTE
# sm = SMOTE(random_state=1)
# data_bert, labels_bert = sm.fit_resample(data_bert, labels_bert)

with open('data/train_RF.npy', 'rb') as f:
    data_bert = np.load(f, allow_pickle=True)
with open('data/labels_RF.npy', 'rb') as f:
    labels_bert = np.load(f, allow_pickle=True)

param_grid = {
    'bootstrap': [True],
    'max_depth': [80, 100, 110],
    'max_features': [2, 3],
    'min_samples_leaf': [1, 3, 5],
    'min_samples_split': [2, 6, 10],
    'n_estimators': [100, 200, 300]
}
# Create a based model
rf = ensemble.RandomForestClassifier(random_state=7)
# Instantiate the grid search model
grid_search = GridSearchCV(estimator=rf, param_grid=param_grid,
                           cv=10, n_jobs=-1, verbose=2)
grid_search.fit(data_bert, labels_bert)

# clf = ensemble.RandomForestClassifier(random_state=7)
# clf.fit(data_bert, labels_bert)
# print(clf.get_params())
# print(clf.)
# dump(clf, 'saved_models/RF_top1_SMOTE_7.joblib')

# with open('data/train_RF.npy', 'wb') as f:
#     np.save(f, data_bert)
# with open('data/labels_RF.npy', 'wb') as f:
#     np.save(f, labels_bert)

# clf = load('saved_models/RFtop1.joblib')

clf = grid_search.best_estimator_

print('independent test 2:')
pre = clf.predict_proba(data_bert_val)
pre = np.asarray(pre)
print('SEN:' + str(sensitivity(labels_bert_val, pre).numpy()))
print('SPE:' + str(specificity(labels_bert_val, pre).numpy()))
print('ACC:' + str(acc(labels_bert_val, pre).numpy()))
print('MCC:' + str(mcc(labels_bert_val, pre).numpy()))
print('AUC:' + str(auc(labels_bert_val, pre).numpy()))

print('independent test 1:')
pre = clf.predict_proba(data_bert_val_1)
pre = np.asarray(pre)
print('ACC:' + str(acc(labels_bert_val_1, pre).numpy()))
