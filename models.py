"""
Models
"""
import logging

from keras.layers import (Input, Conv2D, MaxPooling2D, UpSampling2D, Flatten,
                          Dense)
from keras.models import Model


logger = logging.getLogger(__name__)


def get_cae(img_width, img_height, filter_counts, filter_size, downsampling,
            sup_weight=0.):
    input_img = Input(shape=(img_width, img_height, 1))
    x = input_img
    for f, d in zip(filter_counts, downsampling):
        x = Conv2D(f, filter_size, activation='relu', padding='same')(x)
        x = MaxPooling2D((d, d), padding='same')(x)

    encoded = x

    for f, d in zip(filter_counts[::-1], downsampling[::-1]):
        x = Conv2D(f, filter_size, activation='relu', padding='same')(x)
        x = UpSampling2D((d, d))(x)
    decoded = Conv2D(1, filter_size, activation='sigmoid', padding='same')(x)

    z = Flatten()(encoded)
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
    model = get_cae(256, 256, (8, 8, 8), (3, 3), (2, 2, 2), sup_weight=1.)
    model.summary()
