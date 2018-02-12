"""
Train a Convolutional Network on the pet images dataset and
reuse the encoder for training a CAE.
"""
import os
import argparse
import logging
import json
import pathlib

import numpy as np

from models import get_convnet, get_cae
from data import get_dataset
from fitplot import FitPlot, FitPlotBase


logger = logging.getLogger(__name__)
np.random.seed(123456)


def main():
    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser()
    parser.add_argument('-e', '--exp-name', type=str, default='A')
    parser.add_argument('-fc', '--filter-counts', type=int, nargs='+', default=[32, 32, 64])
    parser.add_argument('-fs', '--filter-size', type=int, nargs='+', default=[3, 3])
    parser.add_argument('-d', '--downsampling', type=int, nargs='+', default=[2, 2, 2])
    parser.add_argument('-i', '--img-side', type=int, default=256)
    parser.add_argument('-t', '--transpose', type=int, default=0)
    parser.add_argument('-sl', '--sup-layers', type=int, nargs='+', default=[16])
    args = parser.parse_args()

    filename = os.path.expanduser("~/plot_cae/{}_config.json".format(
        args.exp_name))
    if os.path.exists(filename):
        logger.info("filename=%s exists, skipping", filename)
    else:
        pathlib.Path(os.path.expanduser("~/plot_cae")).mkdir(exist_ok=True)
        with open(filename, 'w') as fwrite:
            json.dump(vars(args), fwrite)
        # train a conv net
        convnet = get_convnet(args.img_side, args.img_side, args.filter_counts,
                              args.filter_size, args.downsampling,
                              args.sup_layers)
        dataset = get_dataset(args.img_side, args.img_side, 'binary')
        FitPlotBase(convnet['model'], dataset, args.exp_name).run()
        # reuse the encoder for training a CAE
        cae = get_cae(args.img_side, args.img_side, args.filter_counts,
                      args.filter_size, args.downsampling,
                      args.transpose, convnet['encoder'], True)
        dataset = get_dataset(args.img_side, args.img_side, 'input')
        FitPlot(cae['model'], dataset, args.exp_name).run()


if __name__ == '__main__':
    main()
