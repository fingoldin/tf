import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# This model, given a sequence of random integers in the encoder, outputs the first n of those numbers in the decoder

# Number of random integers in the encoder
encoder_timesteps = 10
# Number of integers outputted by the decoder
decoder_timesteps = 2
# Hidden size of the RNN cell
hidden_size = 200
# Training is done on n sequences at a time (batch_size = n)
batch_size = 10
learning_rate = 0.001
epochs = 10000000
clip_value = 1000.0
repeat = 1
seed = 1022
language1 = "data/train.en"
language2 = "data/train.vi"

#os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

tf.set_random_seed(seed)
np.random.seed(seed)

file1 = open(language1, "r").read()
file2 = open(language2, "r").read()

ids_to_words1 = list(set(file1.split()))
ids_to_words2 = list(set(file2.split()))

words_to_ids1 = { word: idx for idx, word in enumerate(ids_to_words1) }
words_to_ids2 = { word: idx for idx, word in enumerate(ids_to_words2) }

sentences1 = file1.split(".")
sentences2 = file2.split(".")

all_words1 = [ [ words_to_ids1[word] for word in sentence.split() ] for sentence in sentences1 ]
all_words2 = [ [ words_to_ids2[word] for word in sentence.split() ] for sentence in sentences2 ]

vocab_size = max(len(ids_to_words1), len(ids_to_words2))

encoder_timesteps = max(all_words1, key = len)
decoder_timesteps = max(all_words2, key = len)

words_dataset = tf.data.Dataset.from_tensor_slices((all_words1, all_words2))
batched_words = words_dataset.apply(tf.contrib.data.batch_and_drop_remainder(batch_size)).repeat(repeat)
iterator = batched_words.make_initializable_iterator()
X, Y = iterator.get_next()

print(words_dataset.output_shapes)

X = tf.one_hot(X, input_size, axis = -1)
Y = tf.one_hot(Y, output_size, axis = -1)

norm = 1

encoder_weights = {
    "Wfh": tf.Variable(tf.random_uniform([ hidden_size, hidden_size ]) / norm),
    "Wfx": tf.Variable(tf.random_uniform([ input_size, hidden_size ]) / norm),
    "bf": tf.Variable(tf.constant(1.0, shape = [ 1, hidden_size ])),

    "Wih": tf.Variable(tf.random_uniform([ hidden_size, hidden_size ]) / norm),
    "Wix": tf.Variable(tf.random_uniform([ input_size, hidden_size ]) / norm),
    "bi": tf.Variable(tf.random_uniform([ 1, hidden_size ]) / norm),
    
    "Wch": tf.Variable(tf.random_uniform([ hidden_size, hidden_size ]) / norm),
    "Wcx": tf.Variable(tf.random_uniform([ input_size, hidden_size ]) / norm),
    "bc": tf.Variable(tf.random_uniform([ 1, hidden_size ]) / norm),
    
    "Woh": tf.Variable(tf.random_uniform([ hidden_size, hidden_size ]) / norm),
    "Wox": tf.Variable(tf.random_uniform([ input_size, hidden_size ]) / norm),
    "bo": tf.Variable(tf.random_uniform([ 1, hidden_size ]) / norm),
    
    "Wy": tf.Variable(tf.random_uniform([ hidden_size, output_size ]) / norm),
    "by": tf.Variable(tf.random_uniform([ 1, output_size ]) / norm)
}

decoder_weights = {
    "Wfh": tf.Variable(tf.random_uniform([ hidden_size, hidden_size ]) / norm),
    "Wfx": tf.Variable(tf.random_uniform([ input_size, hidden_size ]) / norm),
    "bf": tf.Variable(tf.constant(1.0, shape = [ 1, hidden_size ])),

    "Wih": tf.Variable(tf.random_uniform([ hidden_size, hidden_size ]) / norm),
    "Wix": tf.Variable(tf.random_uniform([ input_size, hidden_size ]) / norm),
    "bi": tf.Variable(tf.random_uniform([ 1, hidden_size ]) / norm),
    
    "Wch": tf.Variable(tf.random_uniform([ hidden_size, hidden_size ]) / norm),
    "Wcx": tf.Variable(tf.random_uniform([ input_size, hidden_size ]) / norm),
    "bc": tf.Variable(tf.random_uniform([ 1, hidden_size ]) / norm),
    
    "Woh": tf.Variable(tf.random_uniform([ hidden_size, hidden_size ]) / norm),
    "Wox": tf.Variable(tf.random_uniform([ input_size, hidden_size ]) / norm),
    "bo": tf.Variable(tf.random_uniform([ 1, hidden_size ]) / norm),
    
    "Whh": tf.Variable(tf.random_uniform([ 2 * hidden_size, 2 * hidden_size ]) / norm),
    "Wyh": tf.Variable(tf.random_uniform([ 2 * hidden_size, output_size ]) / norm),
    "bh": tf.Variable(tf.random_uniform([ 1, 2 * hidden_size ]) / norm),
    
    "by": tf.Variable(tf.random_uniform([ 1, output_size ]) / norm),
    
    "Wy": tf.Variable(tf.random_uniform([ hidden_size, output_size ]) / norm),
}

alignment_weights = {
    "Wa": tf.Variable(tf.random_uniform([ hidden_size, hidden_size ]))
}

# Simple RNN cell with tanh, returns h_t (the t'th hidden state)
# where x is the t'th input and h is the (t - 1)'th hidden state
def Cell(x, h, c, weights):
    c_n = tf.matmul(h, weights["Wch"]) + tf.matmul(x, weights["Wcx"]) + weights["bc"]
    f = tf.matmul(h, weights["Wfh"]) + tf.matmul(x, weights["Wfx"]) + weights["bf"]
    i = tf.matmul(h, weights["Wih"]) + tf.matmul(x, weights["Wix"]) + weights["bi"]
    o = tf.matmul(h, weights["Woh"]) + tf.matmul(x, weights["Wox"]) + weights["bo"]

    #if context_vector != None:
    #    c_n += tf.matmul(context_vector, weights["Wcc"])
    #    f += tf.matmul(context_vector, weights["Wfc"])
    #    i += tf.matmul(context_vector, weights["Wic"])
    #    o += tf.matmul(context_vector, weights["Woc"])

    c_n = tf.tanh(c_n)
    f = tf.sigmoid(f)
    i = tf.sigmoid(i)
    o = tf.sigmoid(o)
    
    c = f * c + i * c_n
    h = o * tf.tanh(c)

    return (h, c)

def Context(decoder_state, encoder_states, weights):
    # decoder_state has shape [ batch_size, hidden_size ]
    encoder_states = tf.transpose(encoder_states.stack(), [ 1, 0, 2 ])
    # encoder_states now has shape [ batch_size, encoder_timesteps, hidden_size ]

    # Wa has shape [ hidden_size, hidden_size ]
    temp = tf.reshape(decoder_state, [ batch_size, hidden_size, 1 ])

    scores = tf.reshape(tf.matmul(encoder_states, temp), [ batch_size, 1, encoder_timesteps ])
    alignments = tf.nn.softmax(scores, axis = 2)

    context = tf.reshape(tf.matmul(alignments, encoder_states), [ batch_size, hidden_size ])
    
    return context

# Returns y_t from h_t, usually called right after Cell()
def Prediction(h, context_vector, weights):
    if context_vector != None:
        h_prime = tf.tanh(tf.matmul(tf.concat([ h, context_vector ], axis = 1), weights["Whh"]))

        return tf.matmul(h_prime, weights["Wyh"])
        #return tf.matmul(tf.concat([ h, context_vector ], axis = 1), weights["Wyh"])
    else:
        return tf.matmul(h, weights["Wy"])

# Loop for encoder
# t is the time step, encoder_h is the previous hidden state, and 
# encoder_hidden_states is all the previous hidden states
def encoder_loop_body(t, encoder_h, encoder_c, encoder_hidden_states):
    # Get the next hidden state using x_t and the previous hidden state
    encoder_h, encoder_c = Cell(X[t], encoder_h, encoder_c, encoder_weights)
    # Store the new hidden state
    encoder_hidden_states = encoder_hidden_states.write(t, encoder_h)
    #encoder_cell_states = encoder_cell_states.write(t, encoder_c)

    return [ t + 1, encoder_h, encoder_c, encoder_hidden_states ]

# Initial hidden state is all zeros
encoder_h = tf.zeros([ batch_size, hidden_size ])
encoder_c = tf.zeros([ batch_size, hidden_size ])
encoder_hidden_states = tf.TensorArray(tf.float32, encoder_timesteps, element_shape = [ batch_size, hidden_size ], clear_after_read = False)
#encoder_cell_states = tf.TensorArray(tf.float32, encoder_timesteps, element_shape = [ batch_size, hidden_size ])

_,encoder_h,encoder_c,encoder_hidden_states = tf.while_loop(
                    lambda t, h, c, s: t < encoder_timesteps, encoder_loop_body, 
                    [ tf.constant(0), encoder_h, encoder_c, encoder_hidden_states ])

# Decoder loop body. Similar to encoder_loop_body, except instead of storing
# all the hidden states in a tensor array it stores all the outputs of the 
# network in decoder_outputs. Decoder_outputs has size (t + 1), since the Tensor
# at location 0 is the output of the last timestep of the encoder
def decoder_loop_body(t, total_loss, decoder_h, decoder_c, decoder_outputs, acc):
    context_vector = Context(decoder_h, encoder_hidden_states, alignment_weights)
    # context_vector = None
    # Calculate the output for this time step
    pred = Prediction(decoder_h, context_vector, decoder_weights)
    # Store the output
    decoder_outputs = decoder_outputs.write(t, pred)
    # Calculate the next hidden state, using the previous output as x
    decoder_h, decoder_c = Cell(pred, decoder_h, decoder_c, decoder_weights)
    
    # Now, compute the loss

# labels
    y = Y[t]

    this_loss = tf.losses.softmax_cross_entropy(
                    onehot_labels = y,
                    logits = pred)

    # a is an array of length batch_size, where each element is a 1 if that batch element's prediction is equal to its label,
    # or 0 if it is not
    a = tf.cast(tf.equal(tf.argmax(pred, axis = 1), tf.argmax(y, axis = 1)), tf.int32)

    # Store the accuracy vector for this time step
    acc = acc.write(t, a)

    return [ t + 1, total_loss + this_loss, decoder_h, decoder_c, decoder_outputs, acc ]

total_loss = tf.constant(0.0)
# The first hidden state of the decoder is the last hidden state of the encoder
decoder_h = encoder_h
decoder_c = encoder_c
decoder_outputs = tf.TensorArray(tf.float32, decoder_timesteps, element_shape = [ batch_size, output_size ], clear_after_read = False)
# Accuracy storage
acc = tf.TensorArray(tf.int32, decoder_timesteps, element_shape = (batch_size,))

_,total_loss,_,_,decoder_outputs,acc = tf.while_loop(
                        lambda t, l, h, c, o, a: t < decoder_timesteps, decoder_loop_body, 
                        [ tf.constant(0), total_loss, decoder_h, decoder_c, decoder_outputs, acc ])

# Sum the accuracy vectors across the time steps, then check if the sum is equal to the number of decoder timesteps for each 
# element of the batch (which would indicate an entirely correct prediction), and then find the fraction of elements
# that are accurately predicted
acc = tf.reduce_mean(tf.cast(tf.equal(tf.reduce_sum(acc.stack(), axis = 0), tf.constant(decoder_timesteps, shape = [ batch_size ])), tf.float32))

optimizer = tf.train.GradientDescentOptimizer(learning_rate)
gradients = optimizer.compute_gradients(total_loss)
clipped_gradients = [ (tf.clip_by_value(gradient, -clip_value, clip_value), var) for gradient, var in gradients if gradient != None ]
train = optimizer.apply_gradients(clipped_gradients)

losses = []

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    for i in range(epochs):
        sess.run(train)
        loss = sess.run(total_loss)
        losses.append(loss)
        con = sess.run(clipped_gradients[10])#sess.run(Context(decoder_h, encoder_hidden_states, alignment_weights))
        print("Accuracy, loss after epoch " + str(i + 1) + ": " + str(con) + ", " + str(loss))

    plt.plot(np.arange(epochs), losses)
    plt.show()
