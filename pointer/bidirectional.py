import numpy as np
import tensorflow as tf

# Hyperparameters
hidden_size = 640
batch_size = 50
vocab_size = 10
min_timesteps = 3
max_timesteps = 5
epochs = 1000000
print_epoch = 1
learning_rate = 0.001

norm = np.sqrt(hidden_size)

weights = {
    "W1": tf.Variable(tf.random_normal([ 1, 2 * hidden_size, hidden_size ]) / norm),
    "W2": tf.Variable(tf.random_normal([ hidden_size, hidden_size ]) / norm),
    "v": tf.Variable(tf.random_normal([ 1, hidden_size, 1 ]) / norm)
}

# Of shape [ time_steps x batch_size x vocab_size ]
X_ind = tf.placeholder(tf.int32, shape = [ None, batch_size ])
Y_ind = tf.placeholder(tf.int32, shape = [ None, batch_size ])

encoder_steps = tf.shape(X_ind)[0]
decoder_steps = tf.shape(Y_ind)[0]

X = tf.one_hot(X_ind, vocab_size, axis = -1)
Y = tf.one_hot(Y_ind, encoder_steps, axis = -1)

fwd_cell = tf.contrib.rnn.LSTMCell(hidden_size)
bwd_cell = tf.contrib.rnn.LSTMCell(hidden_size)

decoder_cell = tf.contrib.rnn.LSTMCell(hidden_size)

def encoder_condition(t, f, b, fs, bs):
    return tf.less(t, encoder_steps)

def encoder_loop(t, fwd_state, bwd_state, fwd_hidden_states, bwd_hidden_states):
    _,fwd_state = fwd_cell(X[t], fwd_state)
    _,bwd_state = bwd_cell(X[encoder_steps - t - 1], bwd_state)

    fwd_hidden_states = fwd_hidden_states.write(t, fwd_state[1])
    bwd_hidden_states = bwd_hidden_states.write(encoder_steps - t - 1, bwd_state[1])

    return (t + 1, fwd_state, bwd_state, fwd_hidden_states, bwd_hidden_states)

_,fwd_state,bwd_state,fwd_hidden_states,bwd_hidden_states = tf.while_loop(encoder_condition, encoder_loop,
                                        [ tf.constant(0),
                                        fwd_cell.zero_state(batch_size, tf.float32),
                                        bwd_cell.zero_state(batch_size, tf.float32),
                                        tf.TensorArray(tf.float32, encoder_steps, element_shape = [ batch_size, hidden_size ]),
                                        tf.TensorArray(tf.float32, encoder_steps, element_shape = [ batch_size, hidden_size ]) ])

W1 = tf.tile(weights["W1"], [ encoder_steps, 1, 1 ])

# Shape: [ encoder_steps, hidden_size, 1 ]
v = tf.tile(weights["v"], [ encoder_steps, 1, 1 ])

rang = tf.expand_dims(tf.range(batch_size, dtype = tf.int32), 1)

all_hidden_states = tf.concat([ fwd_hidden_states.stack(), bwd_hidden_states.stack() ], axis = -1)

decoder_state = tf.contrib.rnn.LSTMStateTuple(tf.add(fwd_state[0], bwd_state[0]), tf.add(fwd_state[1], bwd_state[1]))

def decoder_condition(t, l, h, o, p):
    return tf.less(t, decoder_steps)

def decoder_loop(t, total_loss, hidden_state, outputs, prev_output):
    _,hidden_state = decoder_cell(prev_output, hidden_state)
    w_sum = tf.matmul(all_hidden_states, W1) + tf.expand_dims(tf.matmul(hidden_state[1], weights["W2"]), 0)
    ut = tf.matmul(tf.tanh(w_sum), v)
    # ut has shape [ encoder_steps, batch_size, 1 ]

    logits = tf.transpose(tf.reshape(ut, [ encoder_steps, batch_size ]))

    loss = tf.losses.softmax_cross_entropy(Y[t], logits)

    pointers = tf.argmax(ut, axis = 0, output_type = tf.int32)
    prev_output = tf.gather_nd(X, tf.concat([ pointers, rang ], axis = 1))

    #sorted_v, sorted_i = tf.nn.top_k(logits, k = beam_size, sorted = False)

    #log_v = tf.log(tf.reshape(tf.tile(tf.expand_dims(sorted_v, -1), [ 1, 1, beam_size ]), [ batch_size, beam_size * beam_size ]))
    #v_matrix = log_v + tf.tile(last_v, [ 1, beam_size ])
    
    #last_v, temp_i = tf.nn.top_k(v_matrix, k = beam_size, sorted = False)

    #last_i

    outputs = outputs.write(t, tf.reshape(pointers, [ batch_size ]))

    return (t + 1, total_loss + loss, hidden_state, outputs, prev_output)

_,total_loss,_,decoder_outputs,_ = tf.while_loop(decoder_condition, decoder_loop,
                                   [ tf.constant(0), tf.constant(0.0), decoder_state,
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
