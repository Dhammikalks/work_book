import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split


def read_dataset():
    df = pd.read_csv('/home/ros/Desktop/work_book/Neural_Networks/Introduction_to_TensorFlow/Basic/sonar.csv')
    X = df[df.columns[0:60]].values
    y = df[df.columns[60]]

    encoder = LabelEncoder()
    encoder.fit(y)
    y = encoder.transform(y)
    Y = one_hot_encode(y)
    print(X.shape)
    return (X,Y)

def one_hot_encode(labels):
    n_labels = len(labels)
    n_unique_labels = len(np.unique(labels))
    one_hot_encode = np.zeros((n_labels,n_unique_labels))
    one_hot_encode[np.arange(n_labels),labels] = 1
    return one_hot_encode


X, Y = read_dataset()

x, Y = shuffle(X,Y,random_state=1)

train_x, train_y, test_x, test_y = train_test_split(X,Y,train_size=0.2,random_state=415)

print(train_x.shape)
print(train_y.shape)
print(test_x.shape)
print(test_y.shape)

learning_rate = 0.3
training_epochs = 1000
cost_history = np.empty(shape=[1],dtype=float)
n_dim = X.shape
print("n_dim", n_dim)
n_class = 2
model_path = "/home/ros/Desktop/work_book/Neural_Networks/Introduction_to_TensorFlow/Basic/model/"

n_hidden_1 = 60
n_hidden_2 = 60
n_hidden_3 = 60
n_hidden_4 = 60
n_hidden_5 = 60

w = tf.Variable(tf.zeros[n_dim,n_class])
b = tf.Variable(tf.zeros[n_class])

#Input and outputs
x = tf.placeholder(tf.float32, [None,n_class])

y = tf.placeholder(tf.float32, [None,n_class])


def multilayer_perceptorn(x,weights,biases):

    layer_1 = tf.add(tf.matmul(x,weights['h1'],biases['b1']))
    layer_1 = tf.nn.sigmoid(layer_1)

    layer_2 = tf.add(tf.matmul(layer_1,weights['h2'],biases['b2']))
    layer_2 = tf.nn.sigmoid(layer_2)

    layer_3 = tf.add(tf.matmul(layer_2,weights['h3'],biases['b3']))
    layer_3 = tf.nn.sigmoid(layer_3)

    layer_4 = tf.add(tf.matmul(layer_3,weights['h4'],biases['b4']))
    layer_4 = tf.nn.relu(layer_4)

    out_layer = tf.add(tf.matmul(x,weights['out'],biases['out']))
    return out_layer

weights = {
    'h1': tf.Variable(tf.truncated_normal([n_dim,n_hidden_1])),
    'h2': tf.Variable(tf.truncated_normal([n_hidden_1,n_hidden_2])),
    'h3': tf.Variable(tf.truncated_normal([n_hidden_2,n_hidden_3])),
    'h4': tf.Variable(tf.truncated_normal([n_hidden_3,n_hidden_4])),
    'out': tf.Variable(tf.truncated_normal([n_hidden_4,n_class]))
}

biases = {
     'h1': tf.Variable(tf.truncated_normal([n_hidden_1])),
     'h2': tf.Variable(tf.truncated_normal([n_hidden_2])),
     'h3': tf.Variable(tf.truncated_normal([n_hidden_3])),
     'h4': tf.Variable(tf.truncated_normal([n_hidden_4])),
     'out': tf.Variable(tf.truncated_normal([n_class]))
 }
init = tf.global_variables_initializer()

saver = tf.train.Saver()

y = multilayer_perceptorn(x,weights,biases)

cost_function = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=y,logits=y_))
train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost_function)
