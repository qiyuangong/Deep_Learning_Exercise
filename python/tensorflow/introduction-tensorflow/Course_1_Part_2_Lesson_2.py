import tensorflow as tf
import numpy as np
from tensorflow import keras

model = tf.keras.Sequential([keras.layers.Dense(units=1, input_shape=[1])])
model.compile(optimizer='sgd', loss='mean_squared_error')

xs = np.array([-1.0, 0.0, 1.0, 2.0, 3.0, 4.0], dtype=float)
ys = np.array([-3.0, -1.0, 1.0, 3.0, 5.0, 7.0], dtype=float)

# Change epochs=500
model.fit(xs, ys, epochs=500)
new_xs = model.predict(np.random.random_sample((5,)))
print(new_xs)
print(model.predict(new_xs))
# print(model.predict([10.0]))
