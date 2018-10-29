import tensorflow as tf
import numpy as np
import CNNEstimator

tf.logging.set_verbosity(tf.logging.INFO)

epochs = 1
batch_size = 100
lamb = 0.001

dataset = tf.contrib.learn.datasets.load_dataset("mnist")
train_features = dataset.train.images
train_labels = np.asarray(dataset.train.labels, dtype=np.int32)
test_features = dataset.test.images;
test_labels = np.asarray(dataset.test.labels, dtype=np.int32)

train_input_fn = tf.estimator.inputs.numpy_input_fn(
    x = train_features,
    y = train_labels,
    batch_size = batch_size,
    num_epochs = None,
    shuffle = True)

classifier = tf.estimator.Estimator(model_fn = CNNEstimator.model_fn,
             model_dir = "./models/mnist_model", params = { 'lamb': lamb, 'img_size': 28 })

classifier.train(input_fn = train_input_fn, steps = epochs)

test_input_fn = tf.estimator.inputs.numpy_input_fn(
    x = test_features,
    y = test_labels,
    num_epochs = 1,
    shuffle = False)

eval_results = classifier.evaluate(input_fn = test_input_fn)
print(eval_results)
