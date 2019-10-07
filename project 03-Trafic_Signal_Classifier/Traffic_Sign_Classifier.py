#!/bin/python -f
# Load pickled data
import pickle
import random
import numpy as np
import tensorflow as tf
import csv
import cv2
import time

import matplotlib.pyplot as plt
from sklearn.utils import shuffle
from tensorflow.contrib.layers import flatten
# TODO: Fill this in based on where you saved the training and testing data

training_file = '../../resources/traffic-signs-data/train.p'
validation_file= '../../resources/traffic-signs-data/valid.p'
testing_file = '../../resources/traffic-signs-data/test.p'

with open(training_file, mode='rb') as f:
    train = pickle.load(f)
with open(validation_file, mode='rb') as f:
    valid = pickle.load(f)
with open(testing_file, mode='rb') as f:
    test = pickle.load(f)


# Number of training, validation, and test examples, image resolution and color channels,
# and number of classification classes
n_train = len(train['features'])
n_test = len(test['features'])
image_shape = train['features'][0,:,:,:].shape

train_unique, train_counts = np.unique(train['labels'], return_counts='True')
test_unique, test_counts = np.unique(test['labels'], return_counts='True')
n_classes_train = len(train_unique)
n_classes_test = len(test_unique)

print("Number of training examples =", n_train)
print("Number of test examples =", n_test)
print("Image data shape =", image_shape)
print("Number of classes in the training set =", n_classes_train)
print("Number of classes in the test set =", n_classes_test)


grid_m = 3
grid_n = 4
rand_grid = np.random.randint(n_train, size=[grid_m,grid_n])

f0, cell0 = plt.subplots(grid_m, grid_n, figsize=(8,6))
for i in range(grid_m):
    for j in range(grid_n):
        cell0[i, j].imshow(train['features'][rand_grid[i,j]])
        cell0[i, j].axis('off')
        cell0[i, j].set_title('Image {}'.format(rand_grid[i,j]))
plt.show()

train_pcent = np.around(np.array([i/n_train for i in train_counts])*100, 1)
test_pcent = np.around(np.array([np.around(i/n_test, 3) for i in test_counts])*100, 1)

with open('./signnames.csv', newline='') as csvfile:
    csvread = csv.reader(csvfile, delimiter=',')
    signtable = {i[0]:i[1] for i in csvread}
signnames = [signtable[str(train_unique[i])] for i in train_unique]

n_tracks_train = (train_counts / 30).astype(int)
n_tracks_test = (test_counts / 30).astype(int)

class_table = np.array([train_unique, signnames, n_tracks_train, train_counts, train_pcent, n_tracks_test, test_counts, test_pcent]).T

np.set_printoptions(suppress=True, precision=1) # suppress scientific data format for printing because it's not very readable
#print(['Class Id', 'Desc.', '# Tracks Train', '# Train', '% Train', '# Tracks Test', '# Test', '% Test'])
#print(class_table)

f1, cell1 = plt.subplots(figsize=(16,8))
cell1.bar(range(len(n_tracks_train)), n_tracks_train, width=0.9)
cell1.set_ylabel("Number of tracks")
cell1.set_xlabel("Class ID")
cell1.set_xticks(np.arange(0,43,1))
cell1.set_title("Number of tracks per traffic sign category in the training dataset")
for ymaj in cell1.yaxis.get_majorticklocs():
    cell1.axhline(y=ymaj, linewidth=0.5)
plt.show()


# Convert the training, validation, and test datasets from RGB to grayscale.
train_features_gray = np.expand_dims(np.asarray([cv2.cvtColor(img, cv2.COLOR_RGB2GRAY) for img in train['features']]), 3)
val_features_gray = np.expand_dims(np.asarray([cv2.cvtColor(img, cv2.COLOR_RGB2GRAY) for img in valid['features']]), 3)
test_features_gray = np.expand_dims(np.asarray([cv2.cvtColor(img, cv2.COLOR_RGB2GRAY) for img in test['features']]), 3)

#This data is what we're actually going to train and test the model on.
X_train, y_train = train_features_gray / 127.5 - 1, train['labels']
X_val, y_val = val_features_gray / 127.5 - 1, valid['labels']
X_test, y_test = test_features_gray / 127.5 - 1, test['labels']
#..........................................................................shuffling  test data
X_train, y_train = shuffle(X_train, y_train)

#...........................................................................Define a batch normalization function that can distinguish between training and inference.
def batch_normalization(x, is_training, offset, scale, pop_mean, pop_var, layer_type, decay=0.99):
    '''
    Perform batch normalization on the layer passed as 'x'.

    Calls one of two batch normalization functions depending on 'is_training'. If 'is_training' is
    true, calls 'train_normalize', which computes and applies the batch moments during training time
    and uses them to incrementally collect a moving average to compute an estimator of the population
    moments for inference. If 'is_training' is false, calls 'inference_normalize', which performs
    batch normalization during inference time using the population moments computed during training.
    '''
    return tf.cond(is_training,
                   lambda: train_normalize(x, offset, scale, pop_mean, pop_var, layer_type, decay),
                   lambda: inference_normalize(x, pop_mean, pop_var, offset, scale))

def train_normalize(x, offset, scale, pop_mean, pop_var, layer_type, decay):
    epsilon = 1e-4
    if layer_type=='conv':
        batch_mean, batch_var = tf.nn.moments(x, axes=[0, 1, 2])
        train_mean = tf.assign(pop_mean, pop_mean * decay + batch_mean * (1 - decay))
        train_var = tf.assign(pop_var, pop_var * decay + batch_var * (1 - decay))
        with tf.control_dependencies([train_mean, train_var]):
            return tf.nn.batch_normalization(x, batch_mean, batch_var, offset, scale, epsilon)
    elif layer_type=='fc':
        batch_mean, batch_var = tf.nn.moments(x, axes=[0])
        train_mean = tf.assign(pop_mean, pop_mean * decay + batch_mean * (1 - decay))
        train_var = tf.assign(pop_var, pop_var * decay + batch_var * (1 - decay))
        with tf.control_dependencies([train_mean, train_var]):
            return tf.nn.batch_normalization(x, batch_mean, batch_var, offset, scale, epsilon)
    else: raise ValueError("No or unknown layer type given. Supported layer types are convolutional ('conv') and fully connected ('fc').")

def inference_normalize(x, pop_mean, pop_var, offset, scale):
    epsilon = 1e-4
    return tf.nn.batch_normalization(x, pop_mean, pop_var, offset, scale, epsilon)
#..........................................................................tensor network
def Lenet():
    lmbda = 0.01
    lmbda_batch = 0.0005
    sigma = 0.01

    init_learning_rate = 0.002
    global_step = tf.Variable(0, trainable=False)
    learning_rate = tf.train.exponential_decay(init_learning_rate, global_step, decay_steps=800, decay_rate=0.9, staircase=True)

    init_learning_rate_batch = 0.002
    global_step_batch = tf.Variable(0, trainable=False)
    learning_rate_batch = tf.train.exponential_decay(init_learning_rate_batch, global_step_batch, decay_steps=800, decay_rate=0.9, staircase=True)


    #................number of color channels
    n_input_channels = 1
    #................converlutional learyer depths
    layer_depth = {
        'conv1': 108,
        'conv2': 200,
        'fc1': 100,
        'fc2': 43
    }
    #................converlutional leayer filter size
    fsize = {
        '1': 5,
        '2': 5
    }
    #................max pool and stride size
    conv_stride = 1
    pool_k = 2
    #...............flag for stop backpropergation for validation set
    is_training = tf.placeholder(tf.bool)
    #...............Keep prob for dropout. Unused in this model.
    keep_prob = tf.placeholder(tf.float32)
    #..............input output place holders

    X = tf.placeholder(tf.float32, (None, 32, 32, n_input_channels))
    y = tf.placeholder(tf.int32, (None))
    y_one_hot = tf.one_hot(y, 43) #number of outputs


    #.............Generate predetermined random weights in order to initialize both networks identically.
    conv1_W_init = np.random.normal(scale=sigma, size=(fsize['1'], fsize['1'], n_input_channels, layer_depth['conv1']))
    conv2_W_init = np.random.normal(scale=sigma, size=(fsize['2'], fsize['2'], layer_depth['conv1'], layer_depth['conv2']))
    fc1_W_init = np.random.normal(scale=sigma, size=(5*5*layer_depth['conv2'],layer_depth['fc1']))
    fc2_W_init = np.random.normal(scale=sigma, size=(layer_depth['fc1'],layer_depth['fc2']))

    conv1_W_init = conv1_W_init.astype(np.float32)
    conv2_W_init = conv2_W_init.astype(np.float32)
    fc1_W_init = fc1_W_init.astype(np.float32)
    fc2_W_init = fc2_W_init.astype(np.float32)

    # Layer 1: Convolutional. Input = 32x32x1. Output = 28x28x6.
    conv1_W = tf.Variable(conv1_W_init)
    conv1_b = tf.Variable(tf.zeros(layer_depth['conv1']))
    conv1 = tf.nn.conv2d(X, conv1_W, [1,conv_stride,conv_stride,1], padding='VALID') + conv1_b
    conv1 = tf.nn.relu(conv1)

    #batch mode
    conv1_W_batch = tf.Variable(conv1_W_init)
    conv1_beta = tf.Variable(tf.zeros(layer_depth['conv1'])) #Scale for batch normalization. Learnable parameter.
    conv1_gamma = tf.Variable(tf.ones(layer_depth['conv1'])) #Offset for batch normalization. Learnable parameter.
    conv1_pop_mean = tf.Variable(tf.zeros(layer_depth['conv1']), trainable=False) #An estimator of the population mean, estimated over the course of training. Not learnable.
    conv1_pop_var = tf.Variable(tf.ones(layer_depth['conv1']), trainable=False) #An estimator of the population variance, estimated over the course of training. Not learnable.
    conv1_batch = tf.nn.conv2d(X, conv1_W_batch, [1,conv_stride,conv_stride,1], padding='VALID')
    conv1_batch = batch_normalization(conv1_batch, is_training, conv1_beta, conv1_gamma, conv1_pop_mean, conv1_pop_var, layer_type='conv')
    conv1_batch = tf.nn.relu(conv1_batch)

    # Pooling. Input = 28x28x6. Output = 14x14x6.
    conv1 = tf.nn.max_pool(conv1, ksize=[1,pool_k,pool_k,1], strides=[1,pool_k,pool_k,1], padding='VALID')
    conv1_batch = tf.nn.max_pool(conv1_batch, ksize=[1,pool_k,pool_k,1], strides=[1,pool_k,pool_k,1], padding='VALID')


    # Layer 2: Convolutional. Input = 14x14x6. Output = 10x10x16.
    conv2_W = tf.Variable(conv2_W_init)
    conv2_b = tf.Variable(tf.zeros(layer_depth['conv2']))
    conv2 = tf.nn.conv2d(conv1, conv2_W, [1,conv_stride,conv_stride,1], padding='VALID') + conv2_b
    conv2 = tf.nn.relu(conv2)

    #batch
    conv2_W_batch = tf.Variable(conv2_W_init)
    conv2_beta = tf.Variable(tf.zeros(layer_depth['conv2']))
    conv2_gamma = tf.Variable(tf.ones(layer_depth['conv2']))
    conv2_pop_mean = tf.Variable(tf.zeros(layer_depth['conv2']), trainable=False)
    conv2_pop_var = tf.Variable(tf.ones(layer_depth['conv2']), trainable=False)
    conv2_batch = tf.nn.conv2d(conv1_batch, conv2_W_batch, [1,conv_stride,conv_stride,1], padding='VALID')
    batch_m, batch_v = tf.nn.moments(conv2_batch, axes=[0, 1, 2])
    conv2_batch = batch_normalization(conv2_batch, is_training, conv2_beta, conv2_gamma, conv2_pop_mean, conv2_pop_var, layer_type='conv')
    conv2_batch = tf.nn.relu(conv2_batch)

    # Pooling. Input = 10x10x16. Output = 5x5x16.
    conv2 = tf.nn.max_pool(conv2, ksize=[1,pool_k,pool_k,1], strides=[1,pool_k,pool_k,1], padding='VALID')
    conv2_batch = tf.nn.max_pool(conv2_batch, ksize=[1,pool_k,pool_k,1], strides=[1,pool_k,pool_k,1], padding='VALID')

    # Flatten. Input = 5x5x16. Output = 400.
    fc0 = flatten(conv2)
    fc0_batch = flatten(conv2_batch)

    # Layer 3: Fully Connected. Input = 400. Output = 120.
    fc1_W = tf.Variable(fc1_W_init)
    fc1_b = tf.Variable(tf.zeros(layer_depth['fc1']))
    fc1 = tf.matmul(fc0, fc1_W) + fc1_b
    fc1 = tf.nn.relu(fc1)

    #batch
    fc1_W_batch = tf.Variable(fc1_W_init)
    fc1_beta = tf.Variable(tf.zeros(layer_depth['fc1']))
    fc1_gamma = tf.Variable(tf.ones(layer_depth['fc1']))
    fc1_pop_mean = tf.Variable(tf.zeros(layer_depth['fc1']), trainable=False)
    fc1_pop_var = tf.Variable(tf.ones(layer_depth['fc1']), trainable=False)
    fc1_batch = tf.matmul(fc0_batch, fc1_W_batch)
    fc1_batch = batch_normalization(fc1_batch, is_training, fc1_beta, fc1_gamma, fc1_pop_mean, fc1_pop_var, layer_type='fc')
    fc1_batch = tf.nn.relu(fc1_batch)

    # Layer 4: Fully Connected. Input = 84. Output = 43.
    fc2_W = tf.Variable(fc2_W_init)
    fc2_b = tf.Variable(tf.zeros(layer_depth['fc2']))
    logits = tf.matmul(fc1, fc2_W) + fc2_b
    #batch
    fc2_W_batch = tf.Variable(fc2_W_init)
    fc2_b_batch = tf.Variable(tf.zeros(layer_depth['fc2']))
    logits_batch = tf.matmul(fc1_batch, fc2_W_batch) + fc2_b_batch

    #Softmax with cross entropy losslabels=
    loss = tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=y_one_hot)) + lmbda*(tf.nn.l2_loss(conv1_W) + tf.nn.l2_loss(conv2_W) + tf.nn.l2_loss(fc1_W) + tf.nn.l2_loss(fc2_W))
    loss_batch = tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits(logits=logits_batch, labels=y_one_hot)) + lmbda_batch*(tf.nn.l2_loss(conv1_W_batch) + tf.nn.l2_loss(conv2_W_batch) + tf.nn.l2_loss(fc1_W_batch) + tf.nn.l2_loss(fc2_W_batch))

    #Adam minimizer
    training_step = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss, global_step=global_step)
    training_step_batch = tf.train.AdamOptimizer(learning_rate=learning_rate_batch).minimize(loss_batch, global_step=global_step_batch)
    #Prediction accuracy op
    correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(y_one_hot, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    #batch
    correct_prediction_batch = tf.equal(tf.argmax(logits_batch, 1), tf.argmax(y_one_hot, 1))
    accuracy_batch = tf.reduce_mean(tf.cast(correct_prediction_batch, tf.float32))

    return (X,
            y,
            is_training,
            keep_prob,
            training_step,
            training_step_batch,
            accuracy,
            accuracy_batch,
            fc1,
            fc1_batch,
            conv2_beta,
            conv2_gamma,
            conv2_pop_mean,
            conv2_pop_var,
            batch_m,
            batch_v,
            tf.train.Saver()
           )

#Small helper functions to evaluate the model in chunks of `batch_size`.

def evaluate(X_data, y_data):
    num_examples = len(X_data)
    total_accuracy = 0
    sess = tf.get_default_session()
    for offset in range(0, num_examples, batch_size):
        batch_X, batch_y = X_data[offset:offset+batch_size], y_data[offset:offset+batch_size]
        accuracy = sess.run(acc, feed_dict={X: batch_X, y: batch_y, keep_prob: 1.0, is_training: False})
        total_accuracy += (accuracy * len(batch_X))
    return total_accuracy / num_examples

def evaluate_batch(X_data, y_data):
    num_examples = len(X_data)
    total_accuracy = 0
    sess = tf.get_default_session()
    for offset in range(0, num_examples, batch_size):
        batch_X, batch_y = X_data[offset:offset+batch_size], y_data[offset:offset+batch_size]
        accuracy = sess.run(acc_batch, feed_dict={X: batch_X, y: batch_y, keep_prob: 1.0, is_training: False})
        total_accuracy += (accuracy * len(batch_X))
    return total_accuracy / num_examples

epochs = 10
batch_size = 128

tf.reset_default_graph()

#Build the graph
X, y, is_training, keep_prob, training_step, training_step_batch, acc, acc_batch, fc1, fc1_batch, conv2_beta, conv2_gamma, conv2_pop_mean, conv2_pop_var, batch_m, batch_v, saver = Lenet()

#Some lists where we'll save a summary of data of interest
fc1s, fc1s_batch, train_accs, train_accs_batch, val_accs, val_accs_batch = [], [], [], [], [], []
conv2_betas, conv2_gammas, conv2_pop_means, conv2_pop_vars, batch_ms, batch_vs = [], [], [], [], [], []

with tf.Session() as sess:
    start_time = time.time()
    sess.run(tf.global_variables_initializer())
    num_examples = len(X_train)
    k = 0 #Counter for the number of training steps (i.e. number of minit batches processed)

    print("Training...")
    print()
    for i in range(epochs):
        print("EPOCH {} ...".format(i+1))
        X_train, y_train = shuffle(X_train, y_train)
        for offset in range(0, num_examples, batch_size):
            #Create a mini batch
            end = offset + batch_size
            batch_X, batch_y = X_train[offset:end], y_train[offset:end]
            #Run a training step
            sess.run([training_step, training_step_batch], feed_dict={X: batch_X,
                                                                   y: batch_y,
                                                                   keep_prob: 0.5,
                                                                   is_training: True})
            k += 1 #Update the training step counter
            #Run the session with the validation dataset to obtain some variable values
            if k % 50 == 0:
                res = sess.run([fc1,
                                fc1_batch,
                                conv2_beta,
                                conv2_gamma,
                                conv2_pop_mean,
                                conv2_pop_var,
                                batch_m,
                                batch_v],
                               feed_dict={X: X_val,
                                          y: y_val,
                                          keep_prob: 1,
                                          is_training: False})
                fc1s.append(np.mean(res[0],axis=0)) #Record the mean value of fc1 over the entire validation set
                fc1s_batch.append(np.mean(res[1],axis=0)) #Record the mean value of fc1_batch over the entire validation set
                conv2_betas.append(res[2])
                conv2_gammas.append(res[3])
                conv2_pop_means.append(res[4])
                conv2_pop_vars.append(res[5])
                batch_ms.append(res[6])
                batch_vs.append(res[7])
        #Evaluate the accuracy of the last epoch
        training_accuracy = evaluate(X_train, y_train)
        training_accuracy_batch = evaluate_batch(X_train, y_train)
        validation_accuracy = evaluate(X_val, y_val)
        validation_accuracy_batch = evaluate_batch(X_val, y_val)
        train_accs.append(training_accuracy)
        train_accs_batch.append(training_accuracy_batch)
        val_accs.append(validation_accuracy)
        val_accs_batch.append(validation_accuracy_batch)

        print("Number of training steps: {}".format(k))
        print("Training Accuracy = {:.3f}".format(training_accuracy))
        print("Training Accuracy with batch = {:.3f}".format(training_accuracy_batch))
        print("Validation Accuracy = {:.3f}".format(validation_accuracy))
        print("Validation Accuracy with batch = {:.3f}".format(validation_accuracy_batch))
        print("Time elapsed: %s seconds" % round(time.time() - start_time, 0))
        print()

    model_name = 'p0_batch_test_model_01'
    saved_model = saver.save(sess, model_name)
    print("Model saved as {}".format(model_name))

    fc1s, fc1s_batch, train_accs, train_accs_batch, val_accs, val_accs_batch = np.array(fc1s), np.array(fc1s_batch), np.array(train_accs), np.array(train_accs_batch), np.array(val_accs), np.array(val_accs_batch)
    conv2_betas, conv2_gammas, conv2_pop_means, conv2_pop_vars = np.array(conv2_betas), np.array(conv2_gammas), np.array(conv2_pop_means), np.array(conv2_pop_vars)
    batch_ms, batch_vs = np.array(batch_ms), np.array(batch_vs)

    np.save('./{}_fc1s'.format(model_name), fc1s)
    np.save('./{}_fc1s_batch'.format(model_name), fc1s_batch)
    np.save('./{}_train_accs'.format(model_name), train_accs)
    np.save('./{}_train_accs_batch'.format(model_name), train_accs_batch)
    np.save('./{}_val_accs'.format(model_name), val_accs)
    np.save('./{}_val_accs_batch'.format(model_name), val_accs_batch)
    np.save('./{}_conv2_betas'.format(model_name), conv2_betas)
    np.save('./{}_conv2_gammas'.format(model_name), conv2_gammas)
    np.save('./{}_conv2_pop_means'.format(model_name), conv2_pop_means)
    np.save('./{}_conv2_pop_vars'.format(model_name), conv2_pop_vars)
    np.save('./{}_conv2_batch_ms'.format(model_name), batch_ms)
    np.save('./{}_conv2_batch_vs'.format(model_name), batch_vs)
