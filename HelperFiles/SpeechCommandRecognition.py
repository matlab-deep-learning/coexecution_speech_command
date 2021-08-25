# This class is used in the 'Run TF from MATLAB' example

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np

class SpeechCommandRecognition(tf.Module):

    def make_model(self):
        # Define the model
        inputs = keras.Input(shape=(98, 50, 1))

        x = layers.Conv2D(12, 3, strides=1, padding='same')(inputs)
        x = layers.BatchNormalization(axis=3)(x)
        x = layers.Activation('relu')(x)

        x = layers.MaxPool2D(pool_size=3, strides=2, padding='same')(x)

        x = layers.Conv2D(2 * 12, 3, strides=1, padding='same')(x)
        x = layers.BatchNormalization(axis=3)(x)
        x = layers.Activation('relu')(x)

        x = layers.MaxPool2D(pool_size=3, strides=2, padding='same')(x)

        x = layers.Conv2D(4 * 12, 3, strides=1, padding='same')(x)
        x = layers.BatchNormalization(axis=3)(x)
        x = layers.Activation('relu')(x)

        x = layers.MaxPool2D(pool_size=3, strides=2, padding='same')(x)

        x = layers.Conv2D(4 * 12, 3, strides=1, padding='same')(x)
        x = layers.BatchNormalization(axis=3)(x)
        x = layers.Activation('relu')(x)
        x = layers.Conv2D(4 * 12, 3, strides=1, padding='same')(x)
        x = layers.BatchNormalization(axis=3)(x)
        x = layers.Activation('relu')(x)

        x = layers.MaxPool2D(pool_size=(13, 1), strides=(1, 1), padding='valid')(x)

        x = layers.Dropout(rate=.2)(x)

        x = layers.Flatten()(x)
        outputs = layers.Dense(11)(x)

        model = keras.Model(inputs=inputs, outputs=outputs)
        return model

    def loss(self, y, y_):
        loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
        l = loss_object(y_true=y, y_pred=y_)
        return l

    def initializeAcc(self):
        self.epoch_loss_avg = tf.keras.metrics.Mean()
        self.epoch_accuracy = tf.keras.metrics.SparseCategoricalAccuracy()
        
    def __init__(self):
        super(SpeechCommandRecognition, self).__init__()
        self.model = self.make_model()
        lr = tf.Variable(.0003, trainable=False, dtype=tf.float32)
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
        self.initializeAcc()

    def forward(self, x, y, training=False):

        x = np.expand_dims(x, 3)
        if training:
            with tf.GradientTape() as tape:
                z = self.model(x)
                loss_value = self.loss(y, z)
                grads = tape.gradient(loss_value, self.model.trainable_variables)

                self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))

                # Track progress
                self.epoch_loss_avg(loss_value)  # Add current batch loss
                # Compare predicted label to actual label
                self.epoch_accuracy(y, self.model(x))
        else:
            return self.model(x)

    def printAcc(self):
        print("Training loss: {:.3f}, Training accuracy: {:.3%}".format(self.epoch_loss_avg.result(),
                                                      self.epoch_accuracy.result()))
        
    