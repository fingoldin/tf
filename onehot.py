import numpy as np
import tensorflow as tf

text_file = "hello.txt"

all_words = [ s for s in open(text_file, "r").read().lower().split() if s.isalpha() ]
unique_words = set(all_words)

features = {
    'word': all_words
}

word_column = tf.feature_column.categorical_column_with_vocabulary_list('word', unique_words);
word_column = tf.feature_column.indicator_column(word_column)

columns = [ word_column ]

inputs = tf.feature_column.input_layer(features, columns)

var_init = tf.global_variables_initializer()
table_init = tf.tables_initializer()

sess = tf.Session()
sess.run((var_init, table_init))

sess.run(inputs)
