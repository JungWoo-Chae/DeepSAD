import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import cifar10
from .preprocessing import create_semisupervised_setting, load_tfdataset


def load_cifar10(cfg):
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
     
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
    
    y_train[idx] = np.array([[int(x in outlier_classes)] for x in  y_train[idx]])
    y_test = np.array([[int(x in outlier_classes)] for x in  y_test])
    train_data = load_tfdataset(cfg, x_train[idx], y_train[idx], semi_targets)
    test_data = load_tfdataset(cfg, x_test, y_test, np.zeros_like(y_test))
    
    return train_data, test_data