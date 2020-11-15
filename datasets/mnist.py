import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import mnist
from .preprocessing import create_semisupervised_setting, load_tfdataset


def load_mnist(cfg):
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
     
    normal_classes = tuple(cfg['normal_class'])
    known_outlier_classes = tuple(cfg['known_outlier_class'])
    outlier_classes = list(range(0, 10))
    for i in normal_classes: outlier_classes.remove(i) 
    outlier_classes = tuple(outlier_classes)

    ratio_known_normal = cfg['ratio_known_normal']
    ratio_known_outlier = cfg['ratio_known_outlier']
    ratio_pollution = cfg['ratio_pollution']
    
    idx, _, semi_targets = create_semisupervised_setting(y_train, normal_classes,
                                                             outlier_classes, known_outlier_classes,
                                                             ratio_known_normal, ratio_known_outlier, ratio_pollution)
    
#     train_data = tf.data.Dataset.from_tensor_slices((x_train[idx], y_train[idx], semi_targets))
#     train_data = train_data.cache()
#     train_data = train_data.shuffle(4096)
#     train_data = train_data.map(normalize_img, num_parallel_calls=tf.data.experimental.AUTOTUNE)
#     train_data = train_data.batch(cfg['batch_size'])
#     train_data = train_data.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
    
#     test_data = tf.data.Dataset.from_tensor_slices((x_test, y_test, np.zeros_like(y_test)))
#     test_data = test_data.cache()
#     test_data = test_data.shuffle(4096)
#     test_data = test_data.map(normalize_img, num_parallel_calls=tf.data.experimental.AUTOTUNE)
#     test_data = test_data.batch(cfg['batch_size'])
#     test_data = test_data.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
    
    y_train[idx] = np.array([int(x in outlier_classes) for x in  y_train[idx]])
    y_test = np.array([int(x in outlier_classes) for x in  y_test])
    train_data = load_tfdataset(cfg, x_train[idx], y_train[idx], semi_targets)
    test_data = load_tfdataset(cfg, x_test, y_test, np.zeros_like(y_test))
    
    return train_data, test_data