"""
Fit and plot
"""
import os
import logging
import json

import matplotlib.pyplot as plt
import pandas as pd

from keras.utils import plot_model
from keras.callbacks import EarlyStopping


logger = logging.getLogger(__name__)


class FitPlotBase(object):
    def __init__(self, model, dataset, exp_name):
        self.model = model
        self.dataset = dataset
        self.exp_name = exp_name
        # pick first N images for deterministic comparison
        self.x_train, _ = next(self.dataset['train_generator'])
        self.x_test, _ = next(self.dataset['validation_generator'])

    def _plot_model(self):
        filename = os.path.expanduser("~/plot_cae/{}_model.png".format(
            self.exp_name))
        plot_model(self.model, show_shapes=True, to_file=filename)

    def _plot_loss(self, key_to_losses):
        filename = os.path.expanduser("~/plot_cae/{}_loss.png".format(
            self.exp_name))
        pd.DataFrame(data=key_to_losses).plot()
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.savefig(filename)
        plt.close()

        filename = os.path.expanduser("~/plot_cae/{}_loss.json".format(
            self.exp_name))
        with open(filename, 'w') as fwrite:
            json.dump(key_to_losses, fwrite)

    def run(self):
        self.model.summary()
        self._plot_model()
        history = self.model.fit_generator(
            self.dataset['train_generator'],
            validation_data=self.dataset['validation_generator'],
            callbacks=[EarlyStopping()],
            epochs=50)
            # epochs=2, steps_per_epoch=2, validation_steps=2)
        self._plot_loss(history.history)


class FitPlot(FitPlotBase):
    def __init__(self, model, dataset, exp_name):
        super().__init__(model, dataset, exp_name)

    def _plot_rec(self, x_test, decoded_imgs, train):
        img_width, img_height = self.dataset['input_shape'][:2]
        n = 5
        plt.figure(figsize=(10, 4))
        for i in range(n):
            # display original
            ax = plt.subplot(2, n, i + 1)
            plt.imshow(x_test[i].reshape(img_width, img_height))
            plt.gray()
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)

            # display reconstruction
            ax = plt.subplot(2, n, i + n + 1)
            plt.imshow(decoded_imgs[i].reshape(img_width, img_height))
            plt.gray()
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)

        filename = os.path.expanduser("~/plot_cae/{}_rec_train_{}.png".format(
            self.exp_name, train))
        plt.savefig(filename)
        plt.close()

    def run(self):
        super().run()
        self._plot_rec(self.x_train, self.model.predict(self.x_train), True)
        self._plot_rec(self.x_test, self.model.predict(self.x_test), False)
