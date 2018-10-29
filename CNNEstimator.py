import tensorflow as tf

def model_fn(features, labels, mode, params):
    lamb = 0.001
    if 'lamb' in params:
        lamb = params['lamb']

    img_size = 28
    if 'img_size' in params:
        img_size = params['img_size']

    input_layer = tf.reshape(features, [-1, int(img_size), int(img_size), 1])

    conv1 = tf.layers.conv2d(
        inputs = input_layer,
        filters = 32,
        kernel_size = [5, 5],
        padding = "same",
        activation = tf.nn.relu)

    pool1 = tf.layers.max_pooling2d(inputs = conv1, pool_size = [2, 2], strides = 2)

    conv2 = tf.layers.conv2d(
        inputs = pool1,
        filters = 64,
        kernel_size = [5, 5],
        padding = "same",
        activation = tf.nn.relu)

    pool2 = tf.layers.max_pooling2d(inputs = conv2, pool_size = [2, 2], strides = 2)

    pool2_flat = tf.reshape(pool2, [-1, int(img_size / 4) * int(img_size / 4) * 64])

    dense = tf.layers.dense(inputs=pool2_flat, units=1024, activation=tf.nn.relu)

    dropout = tf.layers.dropout(inputs = dense, rate = 0.4, training = (mode == tf.estimator.ModeKeys.TRAIN))

    logits = tf.layers.dense(inputs = dropout, units = 10)
    probs = tf.nn.softmax(logits)
    predicted = tf.argmax(input = logits, axis = 1)

    if mode == tf.estimator.ModeKeys.PREDICT:
        predictions = {
            "classes": predicted,
            "probabilites": probs
        }
        
        return tf.estimator.EstimatorSpec(mode = mode, predictions = predictions)

    onehot_labels = tf.one_hot(indices = tf.cast(labels, tf.int32), depth = 10)
    
    loss = tf.losses.softmax_cross_entropy(onehot_labels = onehot_labels, logits = logits)

    if mode == tf.estimator.ModeKeys.TRAIN:
        optimizer = tf.train.GradientDescentOptimizer(learning_rate = lamb)
        train_op = optimizer.minimize(loss = loss, global_step = tf.train.get_global_step())

        return tf.estimator.EstimatorSpec(mode = mode, loss = loss, train_op = train_op)

    if mode == tf.estimator.ModeKeys.EVAL:
        eval_metric = {
            "accuracy": tf.metrics.accuracy(labels = labels, predictions = predicted)
        }

        return tf.estimator.EstimatorSpec(mode = mode, loss = loss, eval_metric_ops = eval_metric)
