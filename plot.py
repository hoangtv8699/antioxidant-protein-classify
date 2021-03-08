import pickle
from utils.helpers import *

history = pickle.load(open("saved_histories/123 adas/" + "model_0", 'rb'))
plot_loss(history, 1)

