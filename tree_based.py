import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn import tree
from sklearn import ensemble
from sklearn.svm import SVC
import matplotlib.pyplot as plt
from joblib import dump, load
from utils.helpers import *
from utils.iFeature_helper import *

# # Load dataset.
# with open('data/train_data_DCT.npy', 'rb') as f:
#     X = np.load(f, allow_pickle=True)
# with open('data/train_labels_DCT.npy', 'rb') as f:
#     y = np.load(f, allow_pickle=True)
#
# print(X.shape)
# print(y.shape)
# #
# clf = tree.DecisionTreeClassifier()
# clf.fit(X, y)
# dump(clf, 'saved_models/DCT.joblib')

# top 100 feature
clf = load('saved_models/DCT.joblib')
importance = clf.feature_importances_
top100 = np.argsort(importance)[::-1][:100]
print(top100)
print(importance[top100])
# summarize feature importance
for i, v in enumerate(importance):
    print('Feature: %0d, Score: %.5f' % (i, v))
# plot feature importance
color = []
for x in range(len(importance)):
    color.append('b')
color[top100[0]] = 'r'
f = plt.figure()
plt.rc('xtick',labelsize=24)
plt.rc('ytick',labelsize=24)
plt.bar([x for x in range(len(importance))], importance, color=color)
plt.xlabel('Feature number', fontsize=24)
plt.ylabel('Importance level (%)', fontsize=24)
plt.show()
f.savefig('hinh 3.png')
# f.savefig('hinh 3.pdf', dpi=700)


# # reduce 1024xN to 1024 by geting mean
# data_bert, labels = read_bert('data/bert_test/independent_2/', padding="none", maxlen=400)
# x1 = []
# for i in range(len(data_bert)):
#     x2 = []
#     for j in range(len(data_bert[i])):
#         x2.append(np.mean(data_bert[i][j]))
#     x1.append(x2)
#
# x1 = np.asarray(x1)
#
# with open('data/independent_2_data_DCT.npy', 'wb') as f:
#     np.save(f, x1)
# with open('data/independent_2_labels_DCT.npy', 'wb') as f:
#     np.save(f, labels)

# # get top 100 data bert
# x, y = read_bert('data/bert_test/independent_1/', padding="pad_sequence", top=top100, maxlen=400)
#
# with open('data/independent_1_data_bert_first.npy', 'wb') as f:
#     np.save(f, x)
# with open('data/independent_1_labels_bert_first.npy', 'wb') as f:
#     np.save(f, y)

