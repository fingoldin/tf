import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

train_steps = 10
predict_steps = 20
predict_begin = "he"
hidden_size = 1024
epochs = 5000
repeat = 5000
learning_rate = 0.1
file_path = "lotr_clean.txt"
load_path = "model_save/model3"
save_path = "model_save/model3"

file_contents = list(open(file_path).read())

ids_to_letters = list(set(file_contents))
letters_to_ids = { c: i for (i, c) in enumerate(ids_to_letters) }
vocab_size = len(ids_to_letters)

all_letters = [ letters_to_ids[c] for c in file_contents ]

words_dataset = tf.data.Dataset.from_tensor_slices((all_letters[:-1], all_letters[1:]))
batched_words = words_dataset.apply(tf.contrib.data.batch_and_drop_remainder(train_steps)).repeat(repeat)
iterator = batched_words.make_initializable_iterator()
next_batch = iterator.get_next()

X = tf.unstack(tf.one_hot(next_batch[0], vocab_size, axis = -1))
Y = tf.unstack(tf.one_hot(next_batch[1], vocab_size, axis = -1))

Wx = tf.Variable(tf.random_normal([ hidden_size, vocab_size ]))
bh = tf.Variable(tf.random_normal([ hidden_size, 1 ]))
Wh = tf.Variable(tf.random_normal([ hidden_size, hidden_size ]))
Wy = tf.Variable(tf.random_normal([ vocab_size, hidden_size ]))
by = tf.Variable(tf.random_normal([ vocab_size, 1 ]))

y = [None] * train_steps
losses = [None] * train_steps

h_t = tf.zeros([ hidden_size, 1 ])

loss = tf.constant(0.0)

def RNN(x, h):
    h_o = tf.nn.tanh(tf.matmul(Wx, x) + tf.matmul(Wh, h) + bh)
    y_o = tf.matmul(Wy, h_o) + by
    return h_o, y_o

for i in range(train_steps):
    h_t, y[i] = RNN(tf.reshape(X[i], [ vocab_size, 1 ]), h_t)
    loss += tf.losses.softmax_cross_entropy(onehot_labels = tf.reshape(Y[i], [ 1, vocab_size ]), logits = tf.reshape(y[i], [ 1, vocab_size ]))

prediction = [None] * predict_steps

h_p = tf.zeros([ hidden_size, 1])

predict_begin_len = len(predict_begin)

for i in range(predict_begin_len):
    letter = tf.reshape(tf.one_hot([ letters_to_ids[predict_begin[i]] ], vocab_size), [ vocab_size, 1 ])
    h_p, y_p = RNN(letter, h_p)
    
    if i == (predict_begin_len - 1):
        prediction[0] = tf.argmax(tf.nn.softmax(tf.reshape(y_p, [ vocab_size ]))) #tf.multinomial(tf.log(tf.reshape(tf.nn.softmax(y_p, axis = 0), [ 1, vocab_size ])), 1)[0][0]

for i in range(predict_steps - 1):
    letter = tf.reshape(tf.one_hot([ prediction[i] ], vocab_size), [ vocab_size, 1 ])
    h_p, y_p = RNN(letter, h_p)
    prediction[i + 1] = tf.argmax(tf.nn.softmax(tf.reshape(y_p, [ vocab_size ]))) #tf.multinomial(tf.log(tf.reshape(tf.nn.softmax(y_p, axis = 0), [ 1, vocab_size ])), 1)[0][0]

optimizer = tf.train.AdagradOptimizer(learning_rate)
#gradients = optimizer.compute_gradients(loss)
#clipped_gradients = [ (tf.clip_by_value(grad, -clip_val, clip_val), var) for grad, var in gradients ]
#train = optimizer.apply_gradients(clipped_gradients)
train = optimizer.minimize(loss)

errors = []

saver = tf.train.Saver()

with tf.Session() as sess:
    try:
        saver.restore(sess, load_path)
    except tf.errors.NotFoundError:
        print("Could not load, initializing")
        sess.run(tf.global_variables_initializer())
        pass

    sess.run(iterator.initializer)
    
    #print(sess.run(y[3]))

    for i in range(epochs):
        sess.run(train)
        #print(sess.run(X))
        error = sess.run(loss)
        errors.append(error)
        print("Error after epoch " + str(i + 1) + ": " + str(error))

    plt.plot(np.arange(epochs), errors)
    plt.show()
    saver.save(sess, save_path)
    print("Model saved")

    pred = sess.run(prediction)

    print("Generated sentence: " + predict_begin, end="")

    for i in range(predict_steps):
        print(ids_to_letters[pred[i]], end="")

    print("")
