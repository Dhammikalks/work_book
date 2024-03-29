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

test_x,train_x, test_y, train_y = train_test_split(X,Y,train_size=0.2,random_state=415)

print("train_x",train_x.shape)
print("train_y",train_y.shape)
print("test_x",test_x.shape)
print("test_y",test_y.shape)

learning_rate = 0.03
training_epochs = 1000
cost_history = np.empty(shape=[1],dtype=float)
n_dim = X.shape[1]
print("n_dim", n_dim)
n_class = 2
model_path = "/home/ros/Desktop/work_book/Neural_Networks/Introduction_to_TensorFlow/Basic/model/"

n_hidden_1 = 60
n_hidden_2 = 60
n_hidden_3 = 60
n_hidden_4 = 60

x = tf.placeholder(tf.float32, [None, n_dim])
w = tf.Variable(tf.zeros([n_dim, n_class]))
b = tf.Variable(tf.zeros([n_class]))

#Input and outputs
y_ = tf.placeholder(tf.float32, [None, n_class])


def multilayer_perceptorn(x,weights,biases):

    layer_1 = tf.add(tf.matmul(x, weights['h1']),biases['b1'])
    layer_1 = tf.nn.sigmoid(layer_1)

    layer_2 = tf.add(tf.matmul(layer_1,weights['h2']),biases['b2'])
    layer_2 = tf.nn.sigmoid(layer_2)

    layer_3 = tf.add(tf.matmul(layer_2,weights['h3']),biases['b3'])
    layer_3 = tf.nn.sigmoid(layer_3)

    layer_4 = tf.add(tf.matmul(layer_3,weights['h4']),biases['b4'])
    layer_4 = tf.nn.relu(layer_4)

    out_layer = tf.add(tf.matmul(layer_4,weights['out']),biases['out'])
    return out_layer

weights = {
    'h1': tf.Variable(tf.truncated_normal([n_dim,n_hidden_1])),
    'h2': tf.Variable(tf.truncated_normal([n_hidden_1,n_hidden_2])),
    'h3': tf.Variable(tf.truncated_normal([n_hidden_2,n_hidden_3])),
    'h4': tf.Variable(tf.truncated_normal([n_hidden_3,n_hidden_4])),
    'out': tf.Variable(tf.truncated_normal([n_hidden_4,n_class]))
}

biases = {
     'b1': tf.Variable(tf.truncated_normal([n_hidden_1])),
     'b2': tf.Variable(tf.truncated_normal([n_hidden_2])),
     'b3': tf.Variable(tf.truncated_normal([n_hidden_3])),
     'b4': tf.Variable(tf.truncated_normal([n_hidden_4])),
     'out': tf.Variable(tf.truncated_normal([n_class]))
 }
init = tf.global_variables_initializer()

saver = tf.train.Saver()

y = multilayer_perceptorn(x, weights, biases)

cost_function = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=y, labels=y_ ))
train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost_function)



with tf.Session() as sess:
    file_writer = tf.summary.FileWriter('/home/ros/Desktop/work_book/Neural_Networks/Introduction_to_TensorFlow/Basic/graph',sess.graph)
    sess.run(init)
    mse_history = []
    accuracy_history = []
    for epoch in range(training_epochs):
        sess.run(train_step,feed_dict={x: train_x, y_:train_y})
        cost = sess.run(cost_function,feed_dict={x:train_x,y_:train_y})
        cost_history = np.append(cost_history,cost)
        correct_prediction = tf.equal(tf.argmax(y,1),tf.argmax(y_,1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))
        pred_y = sess.run(y,feed_dict={x:test_x})
        mse = tf.reduce_mean(tf.square(pred_y-test_y))
        mse_ = sess.run(mse)
        mse_history.append(mse_)
        accuracy = sess.run(accuracy,feed_dict={x:test_x,y_:test_y})
        accuracy_history.append(accuracy)

        print('epoch : ',epoch, '-', 'cost: ',cost, "- MSE ",mse, " - Train Accuracy: ", accuracy)

    save_path = saver.save(sess,model_path)
    print("Model saved in file: %s" % save_path)

    plt.plot(mse_history, 'r')
    plt.show()
    plt.plot(accuracy_history)
    plt.show()

    correct_prediction = tf.equal(tf.argmax(y, 1),tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))
    print("Test Accuracy : " (sess.run(accuracy,feed_dict={x:test_x,y_:test_y})))

    pred_y = sess.run(y,feed_dict={x:test_x})
    mse = tf.reduce_mean(tf.square(pred_y-test_y))
    print("MSE : %.4f",sess.run(mse))
