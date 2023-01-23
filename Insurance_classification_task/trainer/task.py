from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os
#Importing tensorflow version 1.15
import tensorflow as tf
from tensorflow.python.lib.io import file_io

from trainer import model
from trainer import util




def get_args():
    #hyper-parameters are passed using arguments: learning rate, epochs, batch-size
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--job-dir',
        type=str,
        required=True) #For model exporting and checkpoints
    parser.add_argument(
        '--num-epochs',
        type=int,
        default=2) #For testing purpose epochs is kept at 2, can be increased at run time
    parser.add_argument(
        '--batch-size',
        default=128,
        type=int)
    parser.add_argument(
        '--learning-rate',
        default=.01,
        type=float)
    parser.add_argument(
        '--verbosity',
        choices=['DEBUG', 'ERROR', 'FATAL', 'INFO', 'WARN'],
        default='INFO')
    args, _ = parser.parse_known_args()
    return args




def train_and_evaluate(args):
    #Loads the data from util module
    #Trains and evaluates the model and save it to the stoarge on GCP or local storage

    train_x, train_y, test_x, test_y = util.load_data()

    # dimensions
    num_train_examples, input_dim = train_x.shape
    num_test_examples = test_x.shape[0]

    # Create the Keras Model
    classification_model = model.create_keras_model(
        input_dim=input_dim, learning_rate=args.learning_rate)

    # Pass a numpy array by passing DataFrame.values
    training_dataset = model.input_fn(
        features=train_x.values,
        labels=train_y,
        shuffle=True,
        num_epochs=args.num_epochs,
        batch_size=args.batch_size)

    # Pass a numpy array by passing DataFrame.values
    validation_dataset = model.input_fn(
        features=test_x.values,
        labels=test_y,
        shuffle=False,
        num_epochs=args.num_epochs,
        batch_size=num_test_examples)

    # Setup Learning Rate decay.
    lr_decay_cb = tf.keras.callbacks.LearningRateScheduler(
        lambda epoch: args.learning_rate + 0.02 * (0.5 ** (1 + epoch)),
        verbose=True)

    # Setup TensorBoard callback.
    tensorboard_cb = tf.keras.callbacks.TensorBoard(
        os.path.join(args.job_dir, 'keras_tensorboard'),
        histogram_freq=1)

    # Train model
    classification_model.fit(
        training_dataset,
        steps_per_epoch=int(num_train_examples / args.batch_size),
        epochs=args.num_epochs,
        validation_data=validation_dataset,
        validation_steps=1,
        verbose=1,callbacks=[lr_decay_cb,tensorboard_cb])
    export_path = os.path.join(args.job_dir, 'keras_export')
    tf.keras.experimental.export_saved_model(classification_model, export_path)
    print('Model exported to: {}'.format(export_path))







if __name__ == '__main__':
    #Required if we are testing the model from local laptop, if we are running from notebook instance of the GCP it is not required. 
    #os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = r'E:\UDEMY\GCP\Service_account\gcp-ml-demo1-e29cfb0662dd.json
    args = get_args()
    tf.compat.v1.logging.set_verbosity(args.verbosity)
    classification_model=train_and_evaluate(args)
