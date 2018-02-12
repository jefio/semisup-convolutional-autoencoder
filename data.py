"""
Data
"""
import os

from keras.preprocessing.image import ImageDataGenerator


def get_dataset(img_width, img_height, class_mode):

    train_data_dir = os.path.expanduser('~/data/train')
    validation_data_dir = os.path.expanduser('~/data/validation')
    # 256 memory error
    batch_size = 32

    train_datagen = ImageDataGenerator(rescale=1. / 255)
    test_datagen = ImageDataGenerator(rescale=1. / 255)

    train_generator = train_datagen.flow_from_directory(
        train_data_dir,
        target_size=(img_height, img_width),
        batch_size=batch_size,
        class_mode=class_mode,
        color_mode='grayscale')

    validation_generator = test_datagen.flow_from_directory(
        validation_data_dir,
        target_size=(img_height, img_width),
        batch_size=batch_size,
        class_mode=class_mode,
        color_mode='grayscale')

    return {'train_generator': train_generator,
            'validation_generator': validation_generator,
            'input_shape': (img_width, img_height, 1)}
