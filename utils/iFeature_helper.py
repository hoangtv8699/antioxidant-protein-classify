import numpy as np


def read_iFeature(path):
    f = open(path, 'r')
    labels = []
    data = []
    for line in f:
        line = line.split('  ')
        labels.append(float(line[0]))
        tmp = []
        for i in range(1, len(line)):
            tmp.append(float(line[i].split(':')[1]))
        data.append(tmp)
    return np.asarray(data), np.asarray(labels)


# path = '../data/iFeature/'
# data, labels = read_iFeature(path + 'APAAC.txt')
# new_shape = (data.shape[0], 20, int(data.shape[1] / 20))
# print(data.reshape(new_shape).shape)
# print(labels)
