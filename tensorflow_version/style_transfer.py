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

CONTENT_IMAGE_DIR = "./Data/Experiments/{}/gen_samples".format(args.experiment)
STYLE_IMAGE_DIR = args.style_image_dir
OUTPUT_DIR_ROOT = "./Data/Experiments/{}/styled_gen_samples".format(args.experiment)

def execute_style_transfer(content_image_path, style_image_path, output_image_path):
    params = {
        'content_image' : content_image_path,
        'style_image' : style_image_path,
        'output_image' : output_image_path,
        'image_size' : 450,
        'style_size' : 512,
        'content_layer' : 3,
        'content_weight' : args.content_weight, # 5e-2 by default
        'style_layers' : (1, 4, 6, 7),
        'style_weights' : (2000000, 800, 12, 1),
        'tv_weight' : args.tv_weight # 5e-3 by default
    }
    style_transfer(**params)

def start_style_transfer(content_image_dir, style_image_dir, output_dir_root):
    print("CONTENT IMAGE DIR IS:", content_image_dir)
    if not os.path.exists(output_dir_root):
        os.makedirs(output_dir_root)
    for style_file in os.listdir(style_image_dir):
        if style_file.endswith('.jpg'):
            print("Style transferring for", style_file)
            output_dir_style = os.path.join(output_dir_root, style_file[:-4])
            if not os.path.exists(output_dir_style):
                os.makedirs(output_dir_style)
            for content_file in os.listdir(content_image_dir):
                if content_file.endswith('.jpg'):
                    execute_style_transfer(
                        content_image_path=os.path.join(content_image_dir, content_file),
                        style_image_path=os.path.join(style_image_dir, style_file),
                        output_image_path=os.path.join(output_dir_style, content_file[:-4] + '_' + style_file),
                    )
                elif os.path.isdir(os.path.join(content_image_dir, content_file)):
                    sub_content_image_dir = os.path.join(content_image_dir, content_file)
                    sub_output_dir_root = os.path.join(output_dir_root, content_file)
                    print()
                    print(sub_content_image_dir, sub_output_dir_root)
                    print()
                    start_style_transfer(sub_content_image_dir, style_image_dir, sub_output_dir_root)

print("Start style transfer")
start_style_transfer(CONTENT_IMAGE_DIR, STYLE_IMAGE_DIR, OUTPUT_DIR_ROOT)
