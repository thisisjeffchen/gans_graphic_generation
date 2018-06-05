# Code derived from tensorflow/tensorflow/models/image/imagenet/classify_image.py
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os
import skimage.io
import csv
from Utils.inception_score import get_inception_score


def run_inception_score(data_dir):
    images = []
    for f in os.listdir(data_dir):
        if f.endswith('.jpg'):
            img = skimage.io.imread(os.path.join(data_dir, f))
            images.append(img)

    score = get_inception_score(images) if len(images) > 0 else 0
    return score

def walk_dir(dir, split):
    rows = []
    columns = set(["experiment"])
    for experiment in os.listdir(dir):
        gen_samples = os.path.join(dir, experiment, "{}_samples".format(split))
        if os.path.isdir(gen_samples):
            exper_scores = {"experiment": experiment}
            score = run_inception_score(gen_samples)
            if score > 0:
                exper_scores['latest'] = score
                columns.add('latest')
            for epoch in os.listdir(gen_samples):
                epoch_samples = os.path.join(gen_samples, epoch)
                if os.path.isdir(epoch_samples):
                    exper_scores[epoch] = run_inception_score(epoch_samples)
                    columns.add(epoch)
            rows.append(exper_scores)


    with open('../results/inception_score.csv', 'w') as f:
        w = csv.DictWriter(f, sorted(list(columns)))
        w.writeheader()
        w.writerows(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--split', type=str, default='gen',
                        help='val/gen')

    parser.add_argument('--experiment', type=str, default="default",
                        help='Experiment of dataset')

    parser.add_argument('--epoch', type=int, default=None,
                        help='Epoch of the trained model to load. Defaults to latest checkpoint')

    parser.add_argument('--results', action='store_true',
                        help='uses results folder instead of Experiments')

    parser.add_argument('--walk', action='store_true',
                        help='Traverse directory and record all weights')


    args = parser.parse_args()

    data_dir = "../results" if args.results else "Data/Experiments"

    if args.walk:
        walk_dir(data_dir, args.split)
    else:
        data_dir = os.path.join(data_dir, "{}".format(args.experiment), "{}_samples".format(args.split))
        if args.epoch is not None:
            data_dir = os.path.join(data_dir, "epoch_{}".format(args.epoch))
        else:
            data_dir = os.path.join(data_dir, "latest".format(args.epoch))

        print('Extracting Inception Score')
        score = run_inception_score(data_dir)
        print('\nInception Score: {}'.format(score))







if __name__ == '__main__':
    main()
