import tensorflow as tf
import numpy as np

# 8-million features
x = np.linspace(0.0, 8.0, 8000000) 
# 8-million labels
y = 0.3*x-0.8+np.random.normal(scale=0.25, size=len(x))

# Initial guesses
w_initial = -0.5
b_initial =  1.0 

w = tf.Variable(w_initial) # Parameters
b = tf.Variable(b_initial)

_BATCH = 8
x_placeholder = tf.placeholder(tf.float32, [_BATCH])
y_placeholder = tf.placeholder(tf.float32, [_BATCH])

y_model = w * x_placeholder + b
total_error = tf.reduce_sum((y_placeholder - y_model) ** 2)

optimizer_operation = tf.train.GradientDescentOptimizer(learning_rate=0.001).minimize(total_error) # Does one step

'''
Create operator for initialization.
'''
initializer_operation = tf.global_variables_initializer()

'''
All calculations are done in a session.
'''
with tf.Session() as session:

    session.run(initializer_operation) # Call operator

    _EPOCHS = 100 # Number of "sweeps" across data
    for iteration in range(_EPOCHS):
        random_indices = np.random.randint(len(x), size=_BATCH) # Randomly sample the data
        feed = {
            x_placeholder: x[random_indices],
            y_placeholder: y[random_indices]
        }
        session.run(optimizer_operation, feed_dict=feed) # Call operator
        if iteration % 10 == 0:
            slope, intercept = session.run((w, b)) # Call "m" and "b", which are operators
            print('Slope:', slope, 'Intercept:', intercept)