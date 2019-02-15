import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import time
import tkinter
import sys

epochs = 5000000
axle_size = 0.585
batch_size = 400
sim_steps = 1000
plot_n = 20
v_scale = 4.0
hidden_size = 16
dt_hidden_size = 16
max_dt = 0.02
initial_trans_box = [[ -6.0, 6.0 ], [ -12.0, 0.0 ], [ -1.6, 1.6 ]]
target_vel_box = [ 0.1, 0.5 ]
target = [ 0.0, 0.0, 0.0 ]
loss_weights = [ 1.0, 1.0, 3.0 ]
learning_rate = 0.1
lamb = 100.0
lamb_smooth = 100.0
lamb_vel = 5.0
lamb_len = 0.0#2.0
lamb_fast = 0.0
load_path = "model_save/model_small0"
save_path = "model_save/model_small0"
log_path = "model_log/"

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

layer1 = Dense("layer1", [ 4, hidden_size ])
layer2 = Dense("layer2", [ hidden_size, hidden_size ])
layer3 = Dense("layer3", [ hidden_size, 2 ])

dt_layer1 = Dense("dt_layer1", [ 4, dt_hidden_size ])
dt_layer2 = Dense("dt_layer2", [ dt_hidden_size, dt_hidden_size ])
dt_layer3 = Dense("dt_layer3", [ dt_hidden_size, 1 ], tf.nn.relu)

def model(X):
    out = layer1(X)
    out = layer2(out)
    out = layer3(out)

    return 0.5 * (out + 1.0)

def dt_model(X):
    out = dt_layer1(X)
    #out = dt_layer2(out)
    out = dt_layer3(out)

    return 0.5 * max_dt * (out + 1.0)

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

initial_trans_box_t = np.transpose(np.array(initial_trans_box))

initial_trans = tf.random_uniform([ batch_size, 3 ])
initial_trans = tf.multiply(initial_trans, initial_trans_box_t[1] - initial_trans_box_t[0])
initial_trans = tf.add(initial_trans, initial_trans_box_t[0])

targets = tf.tile(tf.expand_dims(target, 0), [ batch_size, 1 ])
l_weights = tf.expand_dims(tf.nn.softmax(loss_weights), 0)

target_vels = tf.random_uniform([ batch_size, 1 ])
target_vels = tf.multiply(target_vels, target_vel_box[1] - target_vel_box[0])
target_vels = tf.add(target_vels, target_vel_box[0])

#def norm_trans(trans):
#    return (2.0 * (trans - initial_trans_box_t[0]) / (initial_trans_box_t[1] - initial_trans_box_t[0]) - 1.0)

#norm_target_vels = 2.0 * (target_vels - target_vel_box[0]) / (target_vel_box[1] - target_vel_box[0]) - 1.0

dt = tf.reshape(dt_model(tf.concat([ initial_trans, target_vels ], axis = -1)), [batch_size])

def step(t, vel, last_vel, velocities, trans, transforms):
    #p_diff = targets - trans
    #v_diff = target_vels - vel
    #p_metric = tf.less(tf.reduce_sum(p_diff * p_diff, axis = -1), trans_thresh)
    #v_metric = tf.less(tf.where(tf.reduce_sum(vel, axis = -1) != 0.0, tf.reduce_sum(v_diff * v_diff, axis = -1), tf.zeros([ batch_size ])), vel_thresh)
    #mask = tf.cast(tf.logical_not(tf.logical_and(p_metric, v_metric)), tf.float32)
    #mask = tf.tile(tf.expand_dims(mask, -1), [ 1, 2 ])

    vel = model(tf.concat([ trans, target_vels ], axis = -1))# * mask
    last_vel = vel#tf.where(vel == 0.0, last_vel, vel)
    dtrans = update(trans, vel, dt, axle_size)
    trans = trans + dtrans
    velocities = velocities.write(t, vel)
    transforms = transforms.write(t, trans)

    return (t + 1, vel, last_vel, velocities, trans, transforms)

def cond(t, v, lv, vs, tr, trs):
    #p_diff = targets - tr
    #v_diff = target_vels - v
    #p_metric = tf.reduce_mean(tf.reduce_sum(p_diff * p_diff, axis = -1))
    #v_metric = tf.reduce_mean(tf.reduce_sum(v_diff * v_diff, axis = -1))

    return t < sim_steps #tf.cond(tf.logical_and(tf.less(p_metric, trans_thresh), tf.less(v_metric, vel_thresh)), true_fn = lambda: False, false_fn = lambda: t < steps)

_,_,last_vel,velocities,final_transform,transforms = tf.while_loop(cond, step,
                [ tf.constant(0), tf.zeros([ batch_size, 2 ]), tf.zeros([ batch_size, 2 ]), tf.TensorArray(tf.float32, sim_steps, element_shape = [ batch_size, 2 ]),
                  initial_trans, tf.TensorArray(tf.float32, sim_steps, element_shape = [ batch_size, 3 ]) ])

velocities = velocities.stack()
transforms = transforms.stack()

loss_p = lamb * tf.losses.mean_squared_error(targets, final_transform, l_weights)

smooth_error = tf.abs(velocities[:,:,1] - velocities[:,:,0])
loss_smooth = lamb_smooth * tf.reduce_mean(tf.reduce_max(tf.pow(smooth_error, 4.0), axis = 0))

log_velocities = tf.where(velocities > 0.0, tf.log(velocities), tf.constant(-10.0, shape = [ sim_steps, batch_size, 2 ]))
loss_fast = lamb_fast * tf.reduce_mean(tf.reduce_sum(tf.reduce_sum(log_velocities * log_velocities, axis = -1), axis = 0))

vel_error = last_vel - tf.tile(target_vels, [ 1, 2 ])
loss_vel = lamb_vel * tf.reduce_mean(tf.reduce_sum(vel_error * vel_error, axis = -1))

dtransforms = transforms[1:,:,:] - transforms[0:-1,:,:]
loss_len = lamb_len * tf.reduce_mean(tf.reduce_sum(tf.reduce_sum(dtransforms * dtransforms, axis = 0), axis = -1))

loss = loss_p + loss_smooth + loss_vel + loss_len + loss_fast

loss_summary = tf.summary.scalar("Total loss", loss)
loss_p_summary = tf.summary.scalar("Position loss", loss_p)
loss_smooth_summary = tf.summary.scalar("Smooth loss", loss_smooth)
loss_vel_summary = tf.summary.scalar("Final velocity loss", loss_vel)

all_losses_summary = tf.summary.merge([ loss_summary, loss_p_summary, loss_smooth_summary, loss_vel_summary ])

filewriter = tf.summary.FileWriter(log_path, tf.get_default_graph())

train = tf.train.AdagradOptimizer(learning_rate = learning_rate).minimize(loss, global_step = tf.train.get_or_create_global_step())

saver = tf.train.Saver()

with tf.Session() as sess:
    try:
        saver.restore(sess, load_path)
    except tf.errors.NotFoundError:
        print("Could not load, initializing")
        sess.run(tf.global_variables_initializer())
        pass
    
    #print(sess.run(initial_trans))
    #print(sess.run(target_vels))
    #print(sess.run(model([[0.0, -2.0, 0.0, 0.0]])))
    #print(sess.run(layer2(layer1([[0.0, 0.0, 0.0, 0.0]]))))
    print(sess.run(layer3(layer2(layer1([[0.0, -2.0, 0.0, 0.0]])))))
    #print(sess.run(layer1.weights))
    #print(sess.run(layer2.weights))

    with open("data.text", "w") as fd:
        layers = [ sess.run(layer1.weights), sess.run(layer2.weights), sess.run(layer3.weights) ]
        biases = [ sess.run(layer1.biases), sess.run(layer2.biases), sess.run(layer3.biases) ]
       
        fd.write(str(len(layers) + 1) + "\n")
        fd.write(str(len(layers[0])) + " ")
        for i in range(len(layers)):
            fd.write(str(len(layers[i][1])) + "\n")
            print(len(layers[i]))
            print(len(biases[i]))
#            if i == 0:
            for j in range(len(layers[i])):
                for z in range(len(layers[i][j])):
                    fd.write(str(layers[i][j][z]) + " ")
#            else:
#                for z in range(len(layers[i][j])):
#                    for j in range(len(layers[i])):
#                        fd.write(str(layers[i][j][z]) + " ")

            fd.write("\n")
            for j in range(len(biases[i])):
                fd.write(str(biases[i][j]) + " ")
            fd.write("\n")
        fd.write("\n")

    for i in range(epochs):
        _,tvs,tstep,lv,l1,l2,l3,l4,l5,vels,ts,summary = sess.run((train, target_vels, dt, last_vel, loss_p, loss_smooth, loss_vel, loss_len, loss_fast, velocities, transforms, all_losses_summary))

        #print(str(vels[0][0]) + " with " + str(ts[0][0]))
        #print(str(vels[1][0]) + " with " + str(ts[1][0]))
        #print(lv[0:4])
        print("loss_p: " + str(l1) + "  loss_smooth: " + str(l2) + "  loss_vel: " + str(l3) + "  loss_len: " + str(l4) + "  loss_fast: " + str(l5))

        #print("[Epoch " + str(i + 1) + "]  Loss: " + str(l) + "  Final: " + str(ts[loss_steps - 1][0]))
        
        step = tf.train.get_global_step().eval()
        
        filewriter.add_summary(summary, step)
        #filewriter.add_summary(lps, step)
        #filewriter.add_summary(lss, step)
        #filewriter.add_summary(lvs, step)

        if i % 20 == 0:
            saver.save(sess, save_path)
            print("Model saved")

        if i % 30 == 0:
            x = []
            y = []
            arrows = []
            for j in range(plot_n):
                x.append([])
                y.append([])
                arrows.append([])
                
                #print("Target vel: " + str(tvs[j]) + "  Last vel: " + str(lv[j]))

            #print("Dt: " + str(tstep[0:plot_n]))

            for k in range(ts.shape[0]):
                if k % 40 == 0:
                    for j in range(plot_n):
                        x[j].append(ts[k][j][0])
                        y[j].append(ts[k][j][1])
                        arrows[j].append(ts[k][j][2])
                    

            for j in range(plot_n):
                plt.quiver(x[j], y[j], np.sin(arrows[j]), np.cos(arrows[j]), scale = 50.0, color = np.random.random(3))
            
            plt.xlim(-7.0, 7.0)
            plt.ylim(-13.0, 3.0)
            plt.gca().set_aspect("equal", adjustable="box")
            plt.show(block = True)
            #plt.pause(2)
            plt.close()
