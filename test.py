import tensorflow as tf

m = tf.keras.metrics.AUC()
m.update_state([0, 0, 1, 1], [0, 1, 0, 1])
print(m.result().numpy())