# Code derived from tensorflow/tensorflow/models/image/imagenet/classify_image.py
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os
import skimage.io
from Utils.inception_score import get_inception_score
from Utils.image_processing import load_image_inception



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--split', type=str, default='gen',
                        help='val/gen')
    parser.add_argument('--experiment', type=str, default="default",
                        help='Experiment of dataset')
    args = parser.parse_args()


    data_dir = os.path.join("Data", "Experiments", "{}".format(args.experiment), "{}_samples".format(args.split))

    images = []
    for f in os.listdir(data_dir):
        img = skimage.io.imread(os.path.join(data_dir, f))
        images.append(img)

    print('Extracting Inception Score')
    score = get_inception_score(images)
    print('\nInception Score: {}'.format(score))







if __name__ == '__main__':
    main()
