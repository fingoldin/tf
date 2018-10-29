import numpy as np
import tensorflow as tf
import parse
import sys

batch_size = 4096
learning_rate = 0.1
weights = [[ 1.0, 1.0, 1.0 ]]
epochs = 100000

X = np.array([[ 0.0, 0.0 ]], dtype=np.float64)
Y = np.array([[ 1.0, 0.0, 0.0 ]], dtype=np.float64)
avg_Y = np.array([[ 0.0, 0.0, 0.0 ]], dtype=np.float64)

lines = open("logTue").readlines()
prev_r = None
for line in lines:
    r = parse.parse("[ {time} ] {angle} {left_m} {right_m} {left_d} {right_d} {left_r} {right_r}", line)
    if r != None and prev_r != None:
        vl = 0.5 * (float(r["left_m"]) + float(prev_r["left_m"]))
        vr = 0.5 * (float(r["right_m"]) + float(prev_r["right_m"]))
        dtheta = float(r["angle"]) - float(prev_r["angle"])
        dxl = float(r["left_d"]) - float(prev_r["left_d"])
        dxr = float(r["right_d"]) - float(prev_r["right_d"])
        dt = float(r["time"]) - float(prev_r["time"])

        X = np.append(X, [[ vl, vr ]], axis = 0) 
        y = np.array([[ dtheta, dxl, dxr ]]) / dt
        
        pl = dxl / (dt * vl) if vl != 0.0 else 0.0
        pr = dxr / (dt * vr) if vr != 0.0 else 0.0
        pd = (dxl - dxr) / dtheta if dtheta != 0.0 else 0.0

        print(str(pl) + " " + str(pr) + " " + str(pd))

        avg_Y += y
        Y = np.append(Y, y, axis = 0)
    
    prev_r = r

avg_Y = avg_Y / len(Y)
X = 2.0 * X - 1.0
Y = (Y - avg_Y) / (np.amax(Y, 0) - np.amin(Y, 0))

sys.exit()

print(X[1003])
print(Y[1003])

D = tf.Variable(1.0, "axis_width", dtype=tf.float64)
V1 = tf.Variable(1.0, "v1", dtype=tf.float64)
V2 = tf.Variable(1.0, "v2", dtype=tf.float64)

def update(x):
    x_t = tf.transpose(x)

    lv = x_t[0]
    rv = x_t[1]
    
    dtheta = tf.divide(-(rv - lv), tf.abs(D))
    ddl = lv * V1
    ddr = rv * V2

    return tf.stack([ dtheta, ddl, ddr ], axis = 1)

def network(x):
    W1 = tf.get_variable("w1", shape=[2, 128], dtype=tf.float64)
    W2 = tf.get_variable("w2", shape=[128, 3], dtype=tf.float64)
    
    b1 = tf.get_variable("b1", shape=[1, 128], dtype=tf.float64)
    b2 = tf.get_variable("b2", shape=[1, 3], dtype=tf.float64)

    l1 = tf.tanh(tf.matmul(x, W1) + b1)
    l2 = tf.tanh(tf.matmul(l1, W2) + b2)

    return l2

iterator = tf.data.Dataset.from_tensor_slices((X, Y)).batch(batch_size).repeat(epochs).make_initializable_iterator()
batch_x, labels = iterator.get_next()

preds = update(batch_x)
loss = tf.losses.mean_squared_error(labels, preds, weights)

train = tf.train.AdagradOptimizer(learning_rate = learning_rate).minimize(loss)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    sess.run(iterator.initializer)
     
    for i in range(epochs):
        _,l = sess.run((train, loss))

        print("Ecoch " + str(i + 1) + ": " + str(l))

#    print("D: " + str(D.eval()) + " V1: " + str(V1.eval()) + " V2: " + str(V2.eval()))
