import numpy as np
import tensorflow as tf

a = np.array([1, 0, 0, 1])
b = np.array([1, 1, 0, 1])

a = tf.convert_to_tensor(a)
b = tf.convert_to_tensor(b)

c = tf.math.confusion_matrix(a, b)

tf.print(c)