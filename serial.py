import tensorflow as tf

# your dataset

# Features
x = [ 0.00,  1.00,  2.00, 3.00, 4.00, 5.00, 6.00, 7.00]
# Labels 
y = [-0.82, -0.94, -0.12, 0.26, 0.39, 0.64, 1.02, 1.00]

# Initial
w_initial = -0.5
b_initial = 1.0

w = tf.Variable(w_initial)
b = tf.Variable(b_initial)

total_error = 0.0

for t_x, t_y in zip(x, y):
    y_model = w * t_x + b
    total_error += (y_model - t_y) ** 2

optimizer_operation = tf.train.GradientDescentOptimizer(learning_rate=0.001).minimize(total_error)

initializer_operation = tf.global_variables_initializer()

with tf.Session() as session:
    session.run(initializer_operation)
    _EPOCHS = 100 # number of "sweeps" across data
    for iteration in range(_EPOCHS):
        session.run(optimizer_operation) # Call operator
        if iteration % 10 == 0:
            slope, intercept = session.run((w, b))
            print("Slope", slope, 'Intercept:', intercept)