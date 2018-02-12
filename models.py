"""
Models
"""
import logging

from keras.layers import (Input, Conv2D, MaxPooling2D, UpSampling2D, Flatten,
                          Dense, Conv2DTranspose, Dropout)
from keras.models import Model


logger = logging.getLogger(__name__)


def get_convnet(img_width, img_height, filter_counts, filter_size, downsampling,
                sup_layers):
    input_img = Input(shape=(img_width, img_height, 1))
    x = input_img
    for f, d in zip(filter_counts, downsampling):
        x = Conv2D(f, filter_size, activation='relu', padding='same')(x)
        x = MaxPooling2D((d, d), padding='same')(x)

    encoded = x
    encoder = Model(input_img, encoded)

    z = Flatten()(encoded)
    for sup_layer in sup_layers:
        z = Dense(sup_layer)(z)
        z = Dropout(0.5)(z)
    z = Dense(1, activation='sigmoid')(z)

    model = Model(input_img, z)
    model.compile(optimizer='rmsprop', loss='binary_crossentropy',
                  metrics=['accuracy'])
    return {'model': model, 'encoder': encoder}


def get_cae(img_width, img_height, filter_counts, filter_size, downsampling,
            transpose, encoder, freeze_encoder):
    input_img = Input(shape=(img_width, img_height, 1))
    if freeze_encoder:
        for layer in encoder.layers:
            layer.trainable = False
    x = encoder(input_img)

    for f, d in zip(filter_counts[::-1], downsampling[::-1]):
        if transpose:
            x = Conv2DTranspose(f, filter_size, strides=(d, d),
                                activation='relu', padding='same')(x)
        else:
            x = Conv2D(f, filter_size, activation='relu', padding='same')(x)
            x = UpSampling2D((d, d))(x)
    decoded = Conv2D(1, filter_size, activation='sigmoid', padding='same')(x)
    autoencoder = Model(input_img, decoded)
    autoencoder.compile(optimizer='rmsprop', loss='binary_crossentropy')
    return {'model': autoencoder}


def get_semisup_cae(img_width, img_height, filter_counts, filter_size,
                    downsampling, sup_weight, transpose, sup_layers):
    input_img = Input(shape=(img_width, img_height, 1))
    x = input_img
    for f, d in zip(filter_counts, downsampling):
        x = Conv2D(f, filter_size, activation='relu', padding='same')(x)
        x = MaxPooling2D((d, d), padding='same')(x)

    encoded = x

    for f, d in zip(filter_counts[::-1], downsampling[::-1]):
        if transpose:
            x = Conv2DTranspose(f, filter_size, strides=(d, d),
                                activation='relu', padding='same')(x)
        else:
            x = Conv2D(f, filter_size, activation='relu', padding='same')(x)
            x = UpSampling2D((d, d))(x)
    decoded = Conv2D(1, filter_size, activation='sigmoid', padding='same')(x)

    z = Flatten()(encoded)
    for sup_layer in sup_layers:
        z = Dense(sup_layer)(z)
        z = Dropout(0.5)(z)
    z = Dense(1, activation='sigmoid')(z)

    if sup_weight > 0:
        autoencoder = Model(input_img, outputs=[decoded, z])
        autoencoder.compile(optimizer='rmsprop', loss='binary_crossentropy',
                            loss_weights=[1., sup_weight])
    else:
        autoencoder = Model(input_img, decoded)
        autoencoder.compile(optimizer='rmsprop', loss='binary_crossentropy')
    return autoencoder


if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)
    model = get_convnet(256, 256, (8, 8, 8), (3, 3), (2, 2, 2), [64])
    model.summary()
    model = get_cae(256, 256, (8, 8, 8), (3, 3), (2, 2, 2), 1., False, [64])
    model.summary()
