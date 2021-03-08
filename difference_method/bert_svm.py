import numpy as np
from sklearn import tree
from utils.helpers import *
from joblib import dump, load
from matplotlib import pyplot as plt

# clf = load('saved_models/DCT.joblib')
# importance = clf.feature_importances_
# top100 = np.argsort(importance)[::-1][:100]
# print(top100)
#
# data, labels = read_bert('data/bert_train/', padding="pad_sequence", maxlen=400, top=top100)
#
# print(data.shape)
#
# with open('data/train_data_bert.npy', 'wb') as f:
#     np.save(f, data)
# with open('data/train_labels_bert.npy', 'wb') as f:
#     np.save(f, labels)
# #
# data, labels = read_bert('data/bert_test/independent_2/', padding="pad_sequence", maxlen=400, top=top100)
#
# with open('data/independent_2_data_bert.npy', 'wb') as f:
#     np.save(f, data)
# with open('data/independent_2_labels_bert.npy', 'wb') as f:
#     np.save(f, labels)

print(check_missing('../data/csv/'))