#!/bin/python -f

from keras.models import Sequential
from keras.layers.core import Dense, Activation, Flatten
import pickle
import numpy as np
import cv2
import keras
from keras.optimizers import RMSprop
from sklearn.utils import shuffle

training_file = '../../../resources/small_traffic_set/small_train_traffic.p'
testing_file = '../../../resources/small_traffic_set/small_test_traffic.p'
# Create the Sequential model

with open(training_file, mode='rb') as f:
    train = pickle.load(f)

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

# Convert the training, validation, and test datasets from RGB to grayscale.
train_features_gray = np.expand_dims(np.asarray([cv2.cvtColor(img, cv2.COLOR_RGB2GRAY) for img in train['features']]), 3)
test_features_gray = np.expand_dims(np.asarray([cv2.cvtColor(img, cv2.COLOR_RGB2GRAY) for img in test['features']]), 3)

#This data is what we're actually going to train and test the model on.
x_train, y_train = train_features_gray / 127.5 - 1, train['labels']
x_test, y_test = test_features_gray / 127.5 - 1, test['labels']
#..........................................................................shuffling  test data
#x_train = train['features']
#y_train = train['labels']
#x_test = test['features']
#y_test = test['labels']


# convert class vectors to binary class matrices
y_train = y_train -1
y_train = keras.utils.to_categorical(y_train, n_classes_train)
y_test = y_test -1
y_test = keras.utils.to_categorical(y_test,  n_classes_test)


#x_train, y_train = shuffle(x_train, y_train)

model = Sequential()

#1st Layer - Add a flatten layer
model.add(Flatten(input_shape=(32, 32, 1)))

#2nd Layer - Add a fully connected layer
model.add(Dense(128))

#3rd Layer - Add a ReLU activation layer
model.add(Activation('relu'))

#4th Layer - Add a fully connected layer
model.add(Dense(5))

#5th Layer - Add a ReLU activation layer
model.add(Activation('softmax'))


model.summary()
epochs = 100;

model.compile(loss='categorical_crossentropy',
              optimizer=RMSprop(),
              metrics=['accuracy'])

history = model.fit(x_train, y_train,
                    batch_size=1,
                    epochs=epochs,
                    verbose=1,
                    validation_data=(x_test, y_test))
score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])
