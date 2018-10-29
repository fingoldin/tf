import tensorflow as tf

epochs = 5000
lamb = 0.01

x = tf.constant([[1,2], [3,4], [4,3], [4,1]], dtype=tf.float64)
y_true = tf.constant([[0], [0], [2], [4]], dtype=tf.float64)

linear_model = tf.layers.Dense(units=1)
y_pred = linear_model(x)

loss = tf.losses.mean_squared_error(labels=y_true, predictions=y_pred)
optimizer = tf.train.GradientDescentOptimizer(lamb)
train = optimizer.minimize(loss)

sess = tf.Session()
init = tf.global_variables_initializer()
sess.run(init)

writer = tf.summary.FileWriter(".", sess.graph)

for i in range(epochs):
    print("Error at epoch " + str(i) + ": " + str(sess.run((train, loss))[1]))

print("Final prediction: \n" + str(sess.run(y_pred)))
