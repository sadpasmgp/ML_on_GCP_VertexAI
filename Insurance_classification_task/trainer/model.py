
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.python.saved_model import builder as saved_model_builder
from tensorflow.python.saved_model import signature_constants
from tensorflow.python.saved_model import tag_constants
from tensorflow.python.saved_model.signature_def_utils_impl import predict_signature_def


def input_fn(features, labels, shuffle, num_epochs, batch_size):
    #Input function to generate data which will be used for model training.

    if labels is None:
        inputs = features
    else:
        inputs = (features, labels)
    dataset = tf.data.Dataset.from_tensor_slices(inputs)

    if shuffle:
        dataset = dataset.shuffle(buffer_size=len(features))

    # We call repeat after shuffling, to prevent separate epochs from blending together.
    dataset = dataset.repeat(num_epochs)
    dataset = dataset.batch(batch_size)
    return dataset


def create_keras_model(input_dim, learning_rate):
    #Creating a DNN from Keras for classification
    #Returns a model which is ready to be trained

    Dense = tf.keras.layers.Dense
    model = tf.keras.Sequential(
        [
            Dense(125, activation=tf.nn.relu, kernel_initializer='uniform',
                  input_shape=(input_dim,)),
            Dense(75, activation=tf.nn.relu),
            Dense(65, activation=tf.nn.relu),
            Dense(40, activation=tf.nn.relu),
            Dense(25, activation=tf.nn.relu),
            Dense(1, activation=tf.nn.sigmoid) #since it is a binary classification, last layer is sigmoid
        ])

    # using custom optimizer:

    optimizer = tf.keras.optimizers.RMSprop(lr=learning_rate)

    # Compile Keras model
    model.compile(
        loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    return model
