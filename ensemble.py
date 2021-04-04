import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn import tree
from sklearn import ensemble
from sklearn.svm import SVC
from joblib import dump, load
from utils.helpers import *
from utils.iFeature_helper import *

with open('data/independent_2_data_bert_300.npy', 'rb') as f:
    data_bert_val = np.load(f, allow_pickle=True)
with open('data/independent_2_labels_bert_300.npy', 'rb') as f:
    labels_bert_val = np.load(f, allow_pickle=True)

with open('data/independent_1_data_bert_300.npy', 'rb') as f:
    data_bert_val_1 = np.load(f, allow_pickle=True)
with open('data/independent_1_labels_bert_300.npy', 'rb') as f:
    labels_bert_val_1 = np.load(f, allow_pickle=True)

data_bert_val = data_bert_val[:, 0]
data_bert_val_1 = data_bert_val_1[:, 0]

clf = load('saved_models/RF_up_resample_300.joblib')
pre = clf.predict_proba(data_bert_val)
pre = np.asarray(pre)

pre_1 = clf.predict_proba(data_bert_val_1)
pre_1 = np.asarray(pre_1)

test_path = 'data/test/independent_2/'
test_path_1 = 'data/test/independent_1/'

data, labels = read_data(test_path)
data_1, labels_1 = read_data(test_path_1)
data = np.expand_dims(data, axis=-1).astype(np.float32)
data_1 = np.expand_dims(data_1, axis=-1).astype(np.float32)

path = "saved_models/13/"
model_paths = os.listdir(path)
model = []
for model_path in model_paths:
    model.append(tf.keras.models.load_model(path + model_path, compile=False))

vote = voting(model, data)
vote_1 = voting(model, data_1)
# ave = average(model, data)
# med = median(model, data)

ave = []
for i in range(len(pre)):
    tmp = [(pre[i][0] + vote[i][0]) / 2, (pre[i][1] + vote[i][1]) / 2]
    ave.append(tmp)

ave = np.asarray(ave)

sen = sensitivity(labels, ave).numpy()
spe = specificity(labels, ave).numpy()
accc = acc(labels, ave).numpy()
mccc = mcc(labels, ave).numpy()
aucc = auc(labels, ave).numpy()
print('SEN:' + str(sen))
print('SPE:' + str(spe))
print('ACC:' + str(accc))
print('MCC:' + str(mccc))
print('AUC:' + str(aucc))
ave_1 = []
for i in range(len(pre_1)):
    tmp = [pre_1[i][0] + vote_1[i][0], pre_1[i][1] + vote_1[i][1]]
    ave_1.append(tmp)

accc2 = acc(labels_1, ave_1).numpy()
print('ACC:' + str(accc2))

b = [math.floor(sen * 10000) / 10000, math.floor(spe * 10000) / 10000, math.floor(accc * 10000) / 10000,
     math.floor(mccc * 10000) / 10000, math.floor(aucc * 10000) / 10000, math.floor(accc2 * 10000) / 10000]
pd.DataFrame([b]).to_csv('test.csv')