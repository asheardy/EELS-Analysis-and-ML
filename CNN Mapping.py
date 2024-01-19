# -*- coding: utf-8 -*-
"""
Created on Thu Jan 11 13:42:36 2024

@author: Alex Sheardy
This code creates, trains, and tests a convolutional neural network (CNN) for the categorization of vacuum, nanodiamonds, lacey carbon
and multi-walled carbon nanotubes. The training and testing data has been included, and can be opened seperately to see the shape of the
data.
"""

import time
import numpy as np

import matplotlib.pyplot as plt
import graphviz #Only needed to create figure of model structure
import os


np.random.seed(0)

t0 = time.time()

os.chdir("./CNN")
# ene = np.load("ene.npy") #This loads the x values for plotting spectra, but is not used in this code

def min_max_normalize(data):
    min_val = np.min(data, axis=0)
    max_val = np.max(data, axis=0)
    normalized_data = (data - min_val) / (max_val - min_val)
    return normalized_data

#Load and shuffle training data
x_train = np.load("x_train.npy")
y_train = np.load("y_train.npy")

shuf = np.concatenate((x_train, y_train), axis = 1)
np.random.shuffle(shuf)
x_train, y_train = np.split(shuf, [-1], axis = 1)

x_train = x_train.reshape(3276, 1024)
y_train = y_train.reshape(3276,)

#Load and prepare test data
x_test = np.load('ML_data_x_test.npy') 
y_test = np.load('ML_data_y_test.npy')

#Normalize all x data
x_train = min_max_normalize(x_train)
x_test = min_max_normalize(x_test)

del shuf

from keras.layers import Conv1D, Dense, Flatten, Dropout, MaxPooling1D
from keras import regularizers
from keras.utils import to_categorical
from keras.utils.vis_utils import plot_model

#Convert y data to "one hot" for categorization
y_train_one_hot = to_categorical(y_train, num_classes=4)
y_test_one_hot = to_categorical(y_test, num_classes=4)

#Strength of l2 regularizer
l2 = 0.01
        
#Build Model    
from keras.models import Sequential
model = Sequential()
model.add(Conv1D(64, 3, strides = 2, padding = 'same', activation = 'relu', kernel_regularizer=regularizers.l2(l2), input_shape=(1024,1)))
model.add(MaxPooling1D(pool_size = 2))
model.add(Dropout(0.2))
model.add(Conv1D(64, 3, strides = 2, padding = 'same', activation = 'relu', kernel_regularizer=regularizers.l2(l2)))
model.add(MaxPooling1D(pool_size = 2))
model.add(Dropout(0.2))
model.add(Flatten())
model.add(Dense(128, activation = 'relu'))
model.add(Dropout(0.6))
model.add(Dense(4, activation = 'softmax'))


#Define optimizer
from keras.optimizers import Adam
optimizer = Adam(learning_rate=0.0001)

#Define class weights
from sklearn.utils.class_weight import compute_class_weight
class_weights = compute_class_weight(class_weight = 'balanced', classes = np.unique(y_train), y = y_train)

#Compile and train CNN
model.compile(loss = 'categorical_crossentropy', optimizer = optimizer, metrics=['accuracy'])  
history = model.fit(x_train, y_train_one_hot, batch_size = 128, epochs = 250, class_weight=dict(enumerate(class_weights)))
accuracy = model.evaluate(x_train, y_train_one_hot)
print("Train Accuracy: " + str(accuracy[1]))

#Test CNN
y_pred = model.predict(x_test)
test_loss, test_accuracy = model.evaluate(x_test, y_test_one_hot)
print("Test Accuracy: " + str(test_accuracy) + ". Total Calculation time: " + str(time.time()-t0))
y_pred = np.argmax(y_pred, axis=1).reshape((37,37))
plt.imshow(y_pred, interpolation = 'none')
plt.show()


#Below will save a picture of the model structure, needs graphviz
# plot_model(model, to_file='model_plot.png', show_shapes=True, show_layer_names=False, show_layer_activations=True)

#Below will print the counts of the different materials. 0 is vacuum, 1 is ND, 2 is Lacey, 3 is MWCNT
unique_labels, label_counts = np.unique(y_train, return_counts=True)
for label, count in zip(unique_labels, label_counts): print(f"Category {label}: {count} samples")

