"""
Train a Convolutional Autoencoder on the pet images dataset.
"""
import os
import argparse
import logging
import json
import pathlib

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from models import get_cae
from data import get_dataset
from keras.utils import plot_model


logger = logging.getLogger(__name__)
np.random.seed(123456)


class FitPlot(object):
    def __init__(self, exp_name, filter_counts, filter_size, downsampling,
                 sup_weight, img_side):
        self.exp_name = exp_name
        self.filter_counts = filter_counts
        self.filter_size = filter_size
        self.downsampling = downsampling
        self.sup_weight = sup_weight

        self.img_width = img_side
        self.img_height = img_side
        if self.sup_weight > 0:
            class_mode = 'xxy'
        else:
            class_mode = 'input'
        self.dataset = get_dataset(self.img_width, self.img_height, class_mode)
        # pick first N images for deterministic comparison
        self.x_train, _ = next(self.dataset['train_generator'])
        self.x_test, _ = next(self.dataset['validation_generator'])

        self.model = get_cae(self.img_width, self.img_height,
                             self.filter_counts, self.filter_size,
                             self.downsampling, self.sup_weight)

    def _plot_model(self):
        filename = os.path.expanduser("~/plot_cae/{}_model.png".format(
            self.exp_name))
        plot_model(self.model, show_shapes=True, to_file=filename)

    def _plot_rec(self, x_test, decoded_imgs, train):
        n = 8
        plt.figure(figsize=(20, 4))
        for i in range(n):
            # display original
            ax = plt.subplot(2, n, i + 1)
            plt.imshow(x_test[i].reshape(self.img_width, self.img_height))
            plt.gray()
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)

            # display reconstruction
            ax = plt.subplot(2, n, i + n + 1)
            plt.imshow(decoded_imgs[i].reshape(self.img_width, self.img_height))
            plt.gray()
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)

        filename = os.path.expanduser("~/plot_cae/{}_rec_train_{}.png".format(
            self.exp_name, train))
        plt.savefig(filename)
        plt.close()

    def _plot_loss(self, key_to_losses):
        filename = os.path.expanduser("~/plot_cae/{}_loss.png".format(
            self.exp_name))
        pd.DataFrame(data=key_to_losses).plot()
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.savefig(filename)
        plt.close()

    def run(self):
        self.model.summary()
        self._plot_model()
        history = self.model.fit_generator(
            self.dataset['train_generator'],
            validation_data=self.dataset['validation_generator'],
            epochs=50)
        self._plot_loss(history.history)
        if self.sup_weight > 0:
            self._plot_rec(self.x_train, self.model.predict(self.x_train)[0], True)
            self._plot_rec(self.x_test, self.model.predict(self.x_test)[0], False)
        else:
            self._plot_rec(self.x_train, self.model.predict(self.x_train), True)
            self._plot_rec(self.x_test, self.model.predict(self.x_test), False)


def main():
    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser()
    parser.add_argument('-e', '--exp-name', type=str, default='A')
    parser.add_argument('-fc', '--filter-counts', type=int, nargs='+', default=[8, 8, 8])
    parser.add_argument('-fs', '--filter-size', type=int, nargs='+', default=[3, 3])
    parser.add_argument('-d', '--downsampling', type=int, nargs='+', default=[2, 2, 2])
    parser.add_argument('-s', '--sup-weight', type=float, default=0.)
    parser.add_argument('-i', '--img-side', type=int, default=256)
    args = parser.parse_args()

    filename = os.path.expanduser("~/plot_cae/{}_config.json".format(
        args.exp_name))
    if os.path.exists(filename):
        logger.info("filename=%s exists, skipping", filename)
    else:
        pathlib.Path(os.path.expanduser("~/plot_cae")).mkdir(exist_ok=True)
        with open(filename, 'w') as fwrite:
            json.dump(vars(args), fwrite)
        fp = FitPlot(args.exp_name, args.filter_counts, args.filter_size,
                    args.downsampling, args.sup_weight, args.img_side)
        fp.run()


if __name__ == '__main__':
    main()
