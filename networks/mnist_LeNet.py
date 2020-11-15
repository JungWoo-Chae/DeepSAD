import tensorflow as tf
from tensorflow.keras import Model, layers


class MNIST_LeNet(Model):
    def __init__(self, rep_dim=32):
        super(MNIST_LeNet, self).__init__()
        
        self.rep_dim = rep_dim
        self.conv1 = tf.keras.Sequential([layers.Conv2D(8, 5, strides=1, padding='SAME', input_shape=(28, 28, 1)),
                                          layers.BatchNormalization(),
                                          layers.LeakyReLU(),
                                          layers.MaxPool2D()
                                       ])
        self.conv2 = tf.keras.Sequential([layers.Conv2D(4, 5, strides=1, padding='SAME'),
                                          layers.BatchNormalization(),
                                          layers.LeakyReLU(),
                                          layers.MaxPool2D(),
                                          layers.Flatten()
                                       ])

        self.fc1 = layers.Dense(rep_dim)
        
    def call(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.fc1(x)
        return x