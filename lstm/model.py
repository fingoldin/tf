import tensorflow as tf
import os
import numpy as np
import matplotlib.pyplot as plt

hidden_size = 1024
batch_size = 1000
time_steps = 8
#predict_length = 5
repeat = 1000
epochs = 1000
words_path = "data/lotr_words.out"
ids_path = "data/lotr_ids.out"
load_path = "model_save/8timesteps_1024size_1cell_lotsdata2.chk"
save_path = "model_save/8timesteps_1024size_1cell_lotsdata2.chk"
max_words = 8001
save_epoch = 50
lamb = 0.3
predict_epoch = 10
predict_length = 40
num_cells = 1
plot_epoch = 10

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

words_to_ids = {}
ids_to_words = {}

all_words = np.array([], dtype=np.int64)
num_ids = 0

print("begin")

tf.set_random_seed(1022)
np.random.seed(1022)

with open(words_path) as f:
    done = False
    for line in f:
        for x in line.split():
            if all_words.size >= max_words:
                done = True
                break
            all_words = np.append(all_words, int(x))
        if done:
            break

with open(ids_path) as f:
    for line in f:
        el = line.split("~")
        i = int(el[1])

        ids_to_words[i] = el[0]
        words_to_ids[el[0]] = i
        
        if i > (num_ids - 1):
            num_ids = i + 1


print("finished reading data")

out_weights = tf.Variable(tf.random_normal([hidden_size, num_ids]))
out_bias = tf.Variable(tf.random_normal([num_ids]))

#num_words = all_words.size - 1
#indices = tf.reshape(tf.concat([tf.range(num_words, dtype=tf.int64), all_words[:-1]], axis = 0), [ num_words, 2 ])
#one_hot_words = tf.SparseTensor(indices, np.ones(num_words), [ num_words, num_ids ]) #tf.one_hot(all_words[:-1], num_ids, axis = -1)
#one_hot_words_shifted = tf.one_hot(all_words[1:], num_ids, axis = -1)
words_dataset = tf.data.Dataset.from_tensor_slices((all_words[:-1], all_words[1:]))
#batched_words = words_dataset.apply(tf.contrib.data.batch_and_drop_remainder(batch_size))
batched_words = words_dataset.apply(tf.contrib.data.batch_and_drop_remainder(batch_size * time_steps)).repeat(repeat)
iterator = batched_words.make_initializable_iterator()
next_batch = iterator.get_next()

#all_data = tf.unstack(tf.reshape(one_hot_words, [time_steps, 1, num_ids]))

cells = [ tf.contrib.rnn.LSTMCell(hidden_size) for i in range(num_cells) ]
multi_cell = tf.contrib.rnn.MultiRNNCell(cells)

x_reshape = tf.transpose(tf.reshape(next_batch[0], [batch_size, time_steps]))
X = tf.unstack(tf.one_hot(x_reshape, num_ids, axis = -1))
#print(X)
outputs = tf.nn.static_rnn(multi_cell, X, dtype=tf.float32)[0]

#all_labels = tf.unstack(tf.reshape(one_hot_words_shifted, [time_steps, 1, num_ids]))
labels = tf.unstack(tf.transpose(tf.reshape(next_batch[1], [batch_size, time_steps])))
losses = [ tf.losses.sparse_softmax_cross_entropy(labels = labels[i], logits = (tf.matmul(outputs[i], out_weights) + out_bias)) for i in range(time_steps) ]
train_loss = tf.reduce_mean(losses)

train = tf.train.AdagradOptimizer(learning_rate = lamb).minimize(train_loss)

pred_word = tf.placeholder(tf.float32, shape = [ 1, num_ids ])
#print(pred_word)
pred_state = tuple([ (tf.placeholder(tf.float32, shape = [ 1, hidden_size ]), tf.placeholder(tf.float32, shape = [1, hidden_size])) for i in range(num_cells) ])
rnn_pred = tf.nn.static_rnn(multi_cell, [pred_word], pred_state)
prediction = (tf.nn.softmax(tf.matmul(rnn_pred[0][0], out_weights) + out_bias), rnn_pred[1])

saver = tf.train.Saver()

with tf.Session() as sess:
    try:
        saver.restore(sess, load_path)
    except tf.errors.NotFoundError:
        print("Could not load, initializing")
        sess.run(tf.global_variables_initializer())
        pass

    sess.run(iterator.initializer)
    
    #print(sess.run(labels))

    errors = []

#    print(sess.run(indices))
   # for epoch in range(epochs):
    for epoch in range(epochs):
        try:
            sess.run(train)
            err = sess.run(train_loss)
            errors.append(err)
            print("Training error at epoch " + str(epoch) + ": " + str(err))
       
            if epoch % plot_epoch == 0:
                plt.plot(np.arange(epoch + 1), errors)
                plt.show()

            if epoch % save_epoch == 0:
                saver.save(sess, save_path)
                print("Model saved")
        
            if epoch % predict_epoch == 0:
                start_word = "the"
                p_word = np.zeros([1, num_ids], dtype=np.float32)
                #p_outputs = (np.zeros([1, num_ids], dtype=np.float32),) * num_cells
                p_word[0][words_to_ids[start_word]] = 1.0
                p_state = tuple([ (np.zeros([1, hidden_size], dtype=np.float32), np.zeros([1, hidden_size], dtype=np.float32)) for i in range(num_cells) ])
                print("Sample sentence: '" + start_word + " ", end="")
                for i in range(predict_length):
                    p_word, p_state = sess.run(prediction, feed_dict = { pred_word: p_word, pred_state: p_state })
                    #p_word = p_outputs[-1]
                    w_id = np.random.choice(np.arange(num_ids), p = p_word[0])
                    print(ids_to_words[w_id], end=" ")
                print("'")
        except tf.errors.OutOfRangeError:
            #for i in range(predict_length):
            #    printtf.argmax(
            break
