import tensorflow as tf
import numpy as np

# Hyperparameters
hidden_size = 640
batch_size = 50
vocab_size = 30
min_timesteps = 15
max_timesteps = 20
epochs = 1000000
print_epoch = 50
learning_rate = 0.001

norm = np.sqrt(hidden_size)

weights = {
    "W1": tf.Variable(tf.random_normal([ hidden_size, hidden_size ]) / norm),
    "W2": tf.Variable(tf.random_normal([ hidden_size, hidden_size ]) / norm),
    "v": tf.Variable(tf.random_normal([ hidden_size, 1 ]) / norm)
}

# Of shape [ time_steps x batch_size x vocab_size ]
X_ind = tf.placeholder(tf.int32, shape = [ None, batch_size ])
Y_ind = tf.placeholder(tf.int32, shape = [ None, batch_size ])

encoder_steps = tf.shape(X_ind)[0]
decoder_steps = tf.shape(Y_ind)[0]

X = tf.one_hot(X_ind, vocab_size, axis = -1)
Y = tf.one_hot(Y_ind, encoder_steps, axis = -1)

encoder_cell = tf.contrib.rnn.LSTMCell(hidden_size)
decoder_cell = tf.contrib.rnn.LSTMCell(hidden_size)

def encoder_condition(t, h, a):
    return tf.less(t, encoder_steps)

def encoder_loop(t, hidden_state, all_hidden_states):
    _,hidden_state = encoder_cell(X[t], hidden_state)

    all_hidden_states = all_hidden_states.write(t, hidden_state[1])

    return (t + 1, hidden_state, all_hidden_states)

_,encoder_state,encoder_hidden_states = tf.while_loop(encoder_condition, encoder_loop,
                                        [ tf.constant(0),
                                        encoder_cell.zero_state(batch_size, tf.float32),
                                        tf.TensorArray(tf.float32, encoder_steps, element_shape = [ batch_size, hidden_size ]) ])

encoder_hidden_states = encoder_hidden_states.stack()
W1 = tf.tile(tf.expand_dims(weights["W1"], 0), [ encoder_steps, 1, 1 ])

# Shape: [ encoder_steps, hidden_size, 1 ]
v = tf.tile(tf.expand_dims(weights["v"], 0), [ encoder_steps, 1, 1 ])

rang = tf.expand_dims(tf.range(batch_size, dtype = tf.int32), 1)

def decoder_condition(t, l, h, o, p):
    return tf.less(t, decoder_steps)

def decoder_loop(t, total_loss, hidden_state, outputs, prev_output):
    _,hidden_state = decoder_cell(prev_output, hidden_state)
    w_sum = tf.matmul(encoder_hidden_states, W1) + tf.tile(tf.expand_dims(tf.matmul(hidden_state[1], weights["W2"]), 0), [ encoder_steps, 1, 1 ])
    ut = tf.matmul(tf.tanh(w_sum), v)
    # ut has shape [ encoder_steps, batch_size, 1 ]

    logits = tf.transpose(tf.reshape(ut, [ encoder_steps, batch_size ]))

    loss = tf.losses.softmax_cross_entropy(Y[t], logits)

    pointers = tf.argmax(ut, axis = 0, output_type = tf.int32)
    prev_output = tf.gather_nd(X, tf.concat([ pointers, rang ], axis = 1))

    outputs = outputs.write(t, tf.reshape(pointers, [ batch_size ]))

    return (t + 1, total_loss + loss, hidden_state, outputs, prev_output)

_,total_loss,_,decoder_outputs,_ = tf.while_loop(decoder_condition, decoder_loop,
                                   [ tf.constant(0), tf.constant(0.0), encoder_state,
                                   tf.TensorArray(tf.int32, decoder_steps, element_shape = [ batch_size ]),
                                   tf.zeros([ batch_size, vocab_size ]) ]) 

acc = tf.reduce_mean(tf.reduce_min(tf.cast(tf.equal(decoder_outputs.stack(), Y_ind), tf.float32), 0))

train = tf.train.AdamOptimizer(learning_rate = learning_rate).minimize(total_loss)

config = tf.ConfigProto()
config.gpu_options.allow_growth = True

with tf.Session(config = config) as sess:
    sess.run(tf.global_variables_initializer())

    for i in range(epochs):
        time_steps = np.random.randint(max_timesteps - min_timesteps) + min_timesteps

        batch_x = np.random.randint(vocab_size, size = (time_steps, batch_size), dtype = np.int32)
        batch_y = np.argsort(batch_x, axis = 0)

        #print(batch_x)
        #print(batch_y)

        _,loss,accuracy = sess.run((train, total_loss, acc), feed_dict = { X_ind: batch_x, Y_ind: batch_y })

        #x,y = sess.run((X, Y), feed_dict = { X_ind: batch_x, Y_ind: batch_y })

        #print(x)
        #print(y)
        #print("\n")

        if i % print_epoch == 0:
            print("Epoch " + str(i) + "; Loss: " + str(loss) + ", Accuracy: " + str(accuracy))
