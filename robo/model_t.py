import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import time
import tkinter

epochs = 5000000
axle_size = 1.0
batch_size = 1000
loss_steps = 800
loss_dt = 0.01
plot_n = 5
v_scale = 4.0
hidden_size = 128
target = [ 0.0, 0.0, 0.0 ]
target_vel = [ 0.1, 0.1 ]
loss_weights = [ 1.0, 1.0, 2.0 ]
learning_rate = 0.01
lamb = 100.0
lamb_smooth = 800.0
lamb_vel = 1.0
lamb_len = 2.0
lamb_fast = 0.000001
trans_thresh = 0.01
vel_thresh = 0.01
load_path = "model_save/model9"
save_path = "model_save/model9"

class Dense:
    def __init__(self, name, shape, activation = tf.nn.tanh):
        self.shape = shape
        self.name = name
        self.activation = activation
        
        with tf.variable_scope(self.name, reuse = tf.AUTO_REUSE):
            self.weights = tf.get_variable("weights", shape = shape)
            self.biases = tf.get_variable("biases", shape = [ shape[1] ])

    def variables_list(self):
        return [ self.weights, self.biases ]

    def __call__(self, batch):
        return self.activation(tf.matmul(batch, self.weights) + self.biases)

layer1 = Dense("layer1", [ 3, hidden_size ])
layer2 = Dense("layer2", [ hidden_size, 2 ])

def model(X):
    out = layer1(X)
    out = layer2(out)

    return v_scale * 0.5 * (out + 1.0)

def update(trans, vel, dt, d):
    vel_t = tf.transpose(vel)
    lv = vel_t[0]
    rv = vel_t[1]
    dtheta = -dt * (rv - lv) / d
    theta = tf.transpose(trans)[2] + dtheta
    dx = 0.5 * (rv + lv) * tf.sin(theta) * dt
    dy = 0.5 * (rv + lv) * tf.cos(theta) * dt
    
    #lr = vel * dt
    #dtheta = tf.reshape(tf.matmul(lr, [ [-1.0], [1.0] ]), [batch_size]) / d
    #tf.equals(dtheta, tf.zeros([batch_size]))
    #R = tf.transpose(dtheta)[0] / dtheta
    #half = (R + d * 0.5)
    #dx = half * (tf.cos(dtheta) - 1.0)
    #dy = half * tf.sin(dtheta)
    #dtheta = -1.0 * dtheta

    return tf.stack([ dx, dy, dtheta ], axis = 1)

initial_trans = tf.random_uniform([ batch_size, 3 ])
initial_trans = tf.multiply(initial_trans, [ 8.0, 3.0, 2.4 ])
initial_trans = tf.add(initial_trans, [ -4.0, -7.0, -1.2 ])

targets = tf.tile(tf.expand_dims(target, 0), [ batch_size, 1 ])
l_weights = tf.expand_dims(tf.nn.softmax(loss_weights), 0)

target_vels = tf.tile(tf.expand_dims(target_vel, 0), [ batch_size, 1 ])

def predict(mod, steps, dt, d, initial_t):
    def step(t, vel, last_vel, velocities, trans, transforms):
        #p_diff = targets - trans
        #v_diff = target_vels - vel
        #p_metric = tf.less(tf.reduce_sum(p_diff * p_diff, axis = -1), trans_thresh)
        #v_metric = tf.less(tf.where(tf.reduce_sum(vel, axis = -1) != 0.0, tf.reduce_sum(v_diff * v_diff, axis = -1), tf.zeros([ batch_size ])), vel_thresh)
        #mask = tf.cast(tf.logical_not(tf.logical_and(p_metric, v_metric)), tf.float32)
        #mask = tf.tile(tf.expand_dims(mask, -1), [ 1, 2 ])

        vel = mod(trans)# * mask
        last_vel = vel#tf.where(vel == 0.0, last_vel, vel)
        dtrans = update(trans, vel, dt, d)
        trans = trans + dtrans
        velocities = velocities.write(t, vel)
        transforms = transforms.write(t, trans)

        return (t + 1, vel, last_vel, velocities, trans, transforms)

    def cond(t, v, lv, vs, tr, trs):
        #p_diff = targets - tr
        #v_diff = target_vels - v
        #p_metric = tf.reduce_mean(tf.reduce_sum(p_diff * p_diff, axis = -1))
        #v_metric = tf.reduce_mean(tf.reduce_sum(v_diff * v_diff, axis = -1))

        return t < steps #tf.cond(tf.logical_and(tf.less(p_metric, trans_thresh), tf.less(v_metric, vel_thresh)), true_fn = lambda: False, false_fn = lambda: t < steps)

    _,_,last_v,velocities,trans,transforms = tf.while_loop(cond, step,
                [ tf.constant(0), tf.zeros([ batch_size, 2 ]), tf.zeros([ batch_size, 2 ]), tf.TensorArray(tf.float32, steps, element_shape = [ batch_size, 2 ]),
                  initial_t, tf.TensorArray(tf.float32, steps, element_shape = [ batch_size, 3 ]) ])

    return last_v, velocities, transforms, trans

last_vel,velocities,transforms,final_transform = predict(model, loss_steps, loss_dt, axle_size, initial_trans)
velocities = velocities.stack()
transforms = transforms.stack()

loss_p = lamb * tf.losses.mean_squared_error(targets, final_transform, l_weights)
loss_smooth = lamb_smooth * tf.losses.mean_squared_error(velocities[0:-1,:,:], velocities[1:,:,:])

log_velocities = tf.where(velocities > 0.0, tf.log(velocities / v_scale), tf.zeros([ loss_steps, batch_size, 2 ]))
loss_fast = lamb_fast * tf.reduce_mean(tf.reduce_sum(tf.reduce_sum(log_velocities * log_velocities, axis = -1), axis = 0))

vel_error = last_vel - target_vels
loss_vel = lamb_vel * tf.reduce_mean(tf.reduce_sum(vel_error * vel_error, axis = -1))

dtransforms = transforms[1:,:,:] - transforms[0:-1,:,:]
loss_len = lamb_len * tf.reduce_mean(tf.reduce_sum(tf.reduce_sum(dtransforms * dtransforms, axis = 0), axis = -1))

loss = loss_p + loss_smooth + loss_vel + loss_len + loss_fast

train = tf.train.AdagradOptimizer(learning_rate = learning_rate).minimize(loss)

saver = tf.train.Saver()

with tf.Session() as sess:
    try:
        saver.restore(sess, load_path)
    except tf.errors.NotFoundError:
        print("Could not load, initializing")
        sess.run(tf.global_variables_initializer())
        pass
    
    for i in range(epochs):
        _,lv,l1,l2,l3,l4,l5,vels,ts = sess.run((train, last_vel, loss_p, loss_smooth, loss_vel, loss_len, loss_fast, velocities, transforms))

        #print(str(vels[0][0]) + " with " + str(ts[0][0]))
        #print(str(vels[1][0]) + " with " + str(ts[1][0]))
        #print(lv[0:4])
        print("loss_p: " + str(l1) + "  loss_smooth: " + str(l2) + "  loss_vel: " + str(l3) + "  loss_len: " + str(l4) + "  loss_fast: " + str(l5))

        #print("[Epoch " + str(i + 1) + "]  Loss: " + str(l) + "  Final: " + str(ts[loss_steps - 1][0]))
        
        if i % 8 == 0:
            saver.save(sess, save_path)
            print("Model saved")

        if i % 20 == 0:
            x = []
            y = []
            arrows = []
            for j in range(plot_n):
                x.append([])
                y.append([])
                arrows.append([])

            for k in range(ts.shape[0]):
                if k % 10 == 0:
                    for j in range(plot_n):
                        x[j].append(ts[k][j][0])
                        y[j].append(ts[k][j][1])
                        arrows[j].append(ts[k][j][2])
                    
                    print(vels[k][0])

            for j in range(plot_n):
                plt.quiver(x[j], y[j], np.sin(arrows[j]), np.cos(arrows[j]), scale = 50.0, color = np.random.random(3))
            
            plt.xlim(-5, 5)
            plt.ylim(-7, 3)
            plt.gca().set_aspect("equal", adjustable="box")
            plt.show(block = True)
            #plt.pause(7)
            plt.close()
