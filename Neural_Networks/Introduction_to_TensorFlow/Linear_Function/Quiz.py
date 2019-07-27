import tensorflow as tf

n_features = 120
n_labels = 5



def get_weights(n_features,n_labels):
    weights = tf.Variable(tf.truncated_normal((n_features, n_labels)))
    return weights

def get_biases(n_labels):
    bias = tf.Variable(tf.zeros(n_labels))
    return bais

def linear_function(x,W,b):
    return tf.add(tf.matmul(x,W),b)
