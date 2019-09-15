#!/bin/python -f
# Load pickled data
import pickle
import random
import numpy as np
import tensorflow as tf
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

X_train, y_train = train['features'], train['labels']
X_valid, y_valid = valid['features'], valid['labels']
X_test, y_test = test['features'], test['labels']

assert(len(X_train) == len(y_train))
assert(len(X_valid) == len(y_valid))
assert(len(X_test) == len(y_test))

print()
print("Image Shape: {}".format(X_train[0].shape))
print()
print("Training Set:   {} samples".format(len(X_train)))
print("Validation Set: {} samples".format(len(X_valid)))
print("Test Set:       {} samples".format(len(X_test)))

index = random.randint(0, len(X_train))
image = X_train[index].squeeze()
#.......................
#plt.figure(figsize=(1,1))
#plt.imshow(image, cmap="gray")
#plt.show()
#print(y_train[index])
#.......................
X_train, y_train = shuffle(X_train, y_train)


#...........................................tensorflow
EPOCHS = 10
BATCH_SIZE = 128

#...........................................
