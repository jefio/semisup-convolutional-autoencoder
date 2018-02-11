"""
Train a Convolutional Network on the pet images dataset.
"""
import os
import argparse
import logging
import json
import pathlib

from models import get_convnet
from data import get_dataset
from fitplot import FitPlotBase


logger = logging.getLogger(__name__)


def main():
    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser()
    parser.add_argument('-e', '--exp-name', type=str, default='A')
    parser.add_argument('-fc', '--filter-counts', type=int, nargs='+', default=[8, 8, 8])
    parser.add_argument('-fs', '--filter-size', type=int, nargs='+', default=[3, 3])
    parser.add_argument('-d', '--downsampling', type=int, nargs='+', default=[2, 2, 2])
    parser.add_argument('-i', '--img-side', type=int, default=256)
    parser.add_argument('-sl', '--sup-layers', type=int, nargs='+', default=[64])
    args = parser.parse_args()

    filename = os.path.expanduser("~/plot_cae/{}_config.json".format(
        args.exp_name))
    if os.path.exists(filename):
        logger.info("filename=%s exists, skipping", filename)
    else:
        pathlib.Path(os.path.expanduser("~/plot_cae")).mkdir(exist_ok=True)
        with open(filename, 'w') as fwrite:
            json.dump(vars(args), fwrite)
        model = get_convnet(args.img_side, args.img_side, args.filter_counts,
                            args.filter_size, args.downsampling,
                            args.sup_layers)
        dataset = get_dataset(args.img_side, args.img_side, 'binary')
        fp = FitPlotBase(model, dataset, args.exp_name)
        fp.run()


if __name__ == '__main__':
    main()
