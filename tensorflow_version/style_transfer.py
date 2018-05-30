# Place style images (.jpg only) in the ./styles dir
import os
import argparse
import numpy as np

import sys
sys.path.insert(0, './style_transfer_utils')
from style_transfer_utils import style_transfer

parser = argparse.ArgumentParser()

parser.add_argument('--style_image_dir', type=str, default='./styles',
                    help='directory containing content image files (.jpg)')

parser.add_argument('--experiment', type=str, default='default',
                    help='Experiment of dataset')

parser.add_argument('--tv_weight', type=float, default=5e-3,
                    help='tv weight for style transfer')

parser.add_argument('--content_weight', type=float, default=5e-2,
                    help='Experiment of dataset')

args = parser.parse_args()

content_image_dir = "./Data/Experiments/{}/gen_samples".format(args.experiment)
style_image_dir = args.style_image_dir
output_dir_root = "./Data/Experiments/{}/styled_gen_samples".format(args.experiment)

if not os.path.exists(output_dir_root):
    os.makedirs(output_dir_root)

print("Start style transfer")
for style_file in os.listdir(style_image_dir):
    if style_file.endswith('.jpg'):
        print("Style transferring for", style_file)
        output_dir_style = os.path.join(output_dir_root, style_file[:-4])
        if not os.path.exists(output_dir_style):
            os.makedirs(output_dir_style)
        for content_file in os.listdir(content_image_dir):
            if content_file.endswith('.jpg'):
                params = {
                    'content_image' : os.path.join(content_image_dir, content_file),
                    'style_image' : os.path.join(style_image_dir, style_file),
                    'output_image' : os.path.join(output_dir_style, content_file[:-4] + '_' + style_file),
                    'image_size' : 450,
                    'style_size' : 512,
                    'content_layer' : 3,
                    'content_weight' : args.content_weight, # 5e-2 by default
                    'style_layers' : (1, 4, 6, 7),
                    'style_weights' : (2000000, 800, 12, 1),
                    'tv_weight' : args.tv_weight # 5e-3 by default
                }
                style_transfer(**params)
