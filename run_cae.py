"""
Train a Convolutional Autoencoder on the pet images dataset.
"""
import os
import argparse
import logging
import json

import matplotlib.pyplot as plt
import pandas as pd

from models import get_cae
from data import get_dataset
from keras.utils import plot_model


class FitPlot(object):
    def __init__(self, exp_name, filter_counts, filter_size, downsampling,
                 cae):
        self.exp_name = exp_name
        self.filter_counts = filter_counts
        self.filter_size = filter_size
        self.downsampling = downsampling
        self.cae = cae

        self.img_width = 256
        self.img_height = 256
        if self.cae:
            class_mode = 'input'
        self.dataset = get_dataset(self.img_width, self.img_height, class_mode)
        # pick first N images for deterministic comparison
        self.x_train, _ = next(self.dataset['train_generator'])
        self.x_test, _ = next(self.dataset['validation_generator'])

        self.model = get_cae(self.img_width, self.img_height,
                             self.filter_counts, self.filter_size,
                             self.downsampling)

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
        self._plot_rec(self.x_train, self.model.predict(self.x_train), True)
        self._plot_rec(self.x_test, self.model.predict(self.x_test), False)


def main():
    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser()
    parser.add_argument('-e', '--exp-name', type=str, default='A')
    parser.add_argument('-fc', '--filter-counts', type=int, nargs='+', default=[8, 8, 8])
    parser.add_argument('-fs', '--filter-size', type=int, nargs='+', default=[3, 3])
    parser.add_argument('-d', '--downsampling', type=int, nargs='+', default=[2, 2, 2])
    parser.add_argument('-c', '--cae', type=int, default=1, choices=[0, 1])
    args = parser.parse_args()

    filename = os.path.expanduser("~/plot_cae/{}_config.json".format(
        args.exp_name))
    with open(filename, 'w') as fwrite:
        json.dump(vars(args), fwrite)
    fp = FitPlot(args.exp_name, args.filter_counts, args.filter_size,
                 args.downsampling, args.cae)
    fp.run()


if __name__ == '__main__':
    main()
