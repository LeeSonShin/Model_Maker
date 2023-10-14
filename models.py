import numpy as np
import PIL
import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential


def Mobilenet_v3(width, height, channel, class_num):
    model = tf.keras.applications.MobileNetV3Small(
        input_shape=(width, height, channel),
        alpha=0.5,
        minimalistic=False,
        include_top=True,
        weights=None,
        input_tensor=None,
        classes=class_num,
        pooling=None,
        dropout_rate=0.2,
        classifier_activation="softmax",
        include_preprocessing=False
    )
    return model


def Mobilenet_v2(width, height, channel, class_num):
    model = tf.keras.applications.mobilenet_v2.MobileNetV2(
        input_shape=(width, height, channel),
        alpha=0.5,
        include_top=True,
        weights=None,
        input_tensor=None,
        pooling=None,
        classes=class_num,
        classifier_activation='softmax',
    )
    return model

def Mobilenet_v2(width, height, channel, class_num):
    model = tf.keras.applications.mobilenet_v2.MobileNetV2(
        input_shape=(width, height, channel),
        alpha=0.5,
        include_top=True,
        weights=None,
        input_tensor=None,
        pooling=None,
        classes=class_num,
        classifier_activation='softmax',
    )
    return model


def Resnet50(width, height, channel, class_num):
    model = tf.keras.applications.resnet_v2.ResNet50V2(
    include_top=True,
    weights=None,
    input_tensor=None,
    input_shape=(width, height, channel),
    pooling=None,
    classes=class_num,
    classifier_activation='softmax'
    )
    return model

def cnn_example(width, height, channel, class_num):
    model = Sequential([
        layers.Rescaling(1./255, input_shape=(width, height, channel)),
        layers.Conv2D(4, 3, padding='same', activation='relu'),
        layers.Conv2D(4, 3, padding='same', activation='relu'),
        layers.MaxPooling2D(),
        layers.Conv2D(8, 3, padding='same', activation='relu'),
        layers.Conv2D(8, 3, padding='same', activation='relu'),
        layers.MaxPooling2D(),
        layers.Conv2D(16, 3, padding='same', activation='relu'),
        layers.Conv2D(16, 3, padding='same', activation='relu'),
        layers.MaxPooling2D(),
        layers.Conv2D(32, 3, padding='same', activation='relu'),
        layers.Conv2D(32, 3, padding='same', activation='relu'),
        layers.MaxPooling2D(),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dropout(rate=0.8),
        layers.Dense(class_num, activation="softmax")
    ])
    return model



def main():
    model = Resnet50(96,96,1,5)
    model.summary()

if __name__ == "__main__":
    main()
