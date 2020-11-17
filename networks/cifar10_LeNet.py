import tensorflow as tf
from tensorflow.keras import Model, layers


class CIFAR10_LeNet(Model):
    def __init__(self, rep_dim=32):
        super(CIFAR10_LeNet, self).__init__()
        
        self.rep_dim = rep_dim
        self.conv1 = tf.keras.Sequential([layers.Conv2D(32, 5, strides=1, padding='SAME', input_shape=(32, 32, 3)),
                                          layers.BatchNormalization(),
                                          layers.LeakyReLU(),
                                          layers.MaxPool2D()
                                       ])
        self.conv2 = tf.keras.Sequential([layers.Conv2D(64, 5, strides=1, padding='SAME'),
                                          layers.BatchNormalization(),
                                          layers.LeakyReLU(),
                                          layers.MaxPool2D()
                                       ])
        self.conv3 = tf.keras.Sequential([layers.Conv2D(128, 5, strides=1, padding='SAME'),
                                          layers.BatchNormalization(),
                                          layers.LeakyReLU(),
                                          layers.MaxPool2D(),
                                          layers.Flatten()
                                       ])

        self.fc1 = layers.Dense(rep_dim)
        
    def call(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.fc1(x)
        return x

class CIFAR10_LeNet_decoder(Model):
    def __init__(self, rep_dim=32):
        super(CIFAR10_LeNet_decoder, self).__init__()
        
        self.rep_dim = rep_dim
        self.deconv1 = tf.keras.Sequential([layers.Reshape((4, 4, int(self.rep_dim / (4 * 4)))),
                                          layers.Conv2DTranspose(128, 5, strides=1, padding='SAME'),
                                          layers.BatchNormalization(),
                                          layers.LeakyReLU()
                                       ])
        self.deconv2 = tf.keras.Sequential([layers.Conv2DTranspose(64, 5, strides=2, padding='SAME'),
                                        layers.BatchNormalization(),
                                       layers.LeakyReLU()
                                       ])
        self.deconv3 = tf.keras.Sequential([layers.Conv2DTranspose(32, 5, strides=2, padding='SAME'),
                                        layers.BatchNormalization(),
                                       layers.LeakyReLU()
                                       ])
        self.deconv4 = layers.Conv2DTranspose(3, 5, strides=2, padding='SAME')
        

    def call(self, x):
        x = self.deconv1(x)
        x = self.deconv2(x)
        x = self.deconv3(x)
        x = self.deconv4(x)
        x = tf.nn.sigmoid(x)
        return x