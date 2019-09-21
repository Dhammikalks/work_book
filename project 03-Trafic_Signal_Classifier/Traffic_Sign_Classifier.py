#!/bin/python -f
# Load pickled data
import pickle
import random
import numpy as np
import tensorflow as tf
import csv
import cv2

import matplotlib.pyplot as plt
from sklearn.utils import shuffle
from tensorflow.contrib.layers import flatten
# TODO: Fill this in based on where you saved the training and testing data

training_file = '../../../resources/traffic-signs-data/train.p'
validation_file= '../../../resources/traffic-signs-data/valid.p'
testing_file = '../../../resources/traffic-signs-data/test.p'

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
#..........................................................................
