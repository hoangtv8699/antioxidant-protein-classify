from utils.helpers import *

data, label = read_fasta('../data/training.fasta', maxlen=400, encode='token')

print(data[0])
data = encodes_amino_feature(data)
print(data[0])


