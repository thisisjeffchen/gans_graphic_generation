# Execute this post-processing script to remove whitespace in the segmented images
import os
import argparse
import numpy as np

import sys
sys.path.insert(0, './style_transfer_utils')
from style_transfer_utils import style_transfer


CONTENT_IMAGE_DIR = "./results/20180528_elephant_segmented_07_epochs_1000_gen_updates_10_after_300_epochs/gen_samples"
STYLE_IMAGE_DIR = "./styles"

parser = argparse.ArgumentParser()
parser.add_argument('--content_image_dir', type=str, default=CONTENT_IMAGE_DIR,
                    help='directory containing content image files (.jpg)')

parser.add_argument('--style_image_dir', type=str, default=STYLE_IMAGE_DIR,
                    help='directory containing content image files (.jpg)')

args = parser.parse_args()

content_image_dir = args.content_image_dir
style_image_dir = args.style_image_dir


# takes the content image dir and preps the suffix path for output
# e.g. "./results/gan_output" --> "./results/styled_gan_output"
output_dir_root = '/'.join(content_image_dir.split('/')[:-1]) + '/' + 'styled_' + content_image_dir.split('/')[-1]

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
                    'output_image' : os.path.join(output_dir_style, style_file[:-4] + '_' + content_file),
                    'image_size' : 450,
                    'style_size' : 512,
                    'content_layer' : 3,
                    'content_weight' : 5e-2,
                    'style_layers' : (1, 4, 6, 7),
                    'style_weights' : (2000000, 800, 12, 1),
                    'tv_weight' : 5e-3
                }
                style_transfer(**params)
                print("-- finished transfering:", style_file, "to", content_file)


"""

MSCOCO_RAW_DIR = "Data/mscoco_raw/"
PROCESSED_DIR = os.path.join(MSCOCO_RAW_DIR, "processed")
OUTPUT_DIR = os.path.join(MSCOCO_RAW_DIR, "processed_cleaned")
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

# percentage of pixels that are not white
THRESHOLD = 0.01

num_processed = 0
num_excluded = 0
for filename in os.listdir(PROCESSED_DIR):
  if filename.endswith('.jpg'):
    img = cv2.imread(os.path.join(PROCESSED_DIR, filename), cv2.IMREAD_GRAYSCALE)
    num_white = np.sum(img == 255)
    num_total = img.shape[0] * img.shape[1]
    percent_subject = 1 - (num_white / num_total)
    num_processed += 1

    if percent_subject >= THRESHOLD:
        copyfile(os.path.join(PROCESSED_DIR, filename), os.path.join(OUTPUT_DIR, filename))
    else:
        num_excluded += 1

    if num_processed % 1000 == 0:
        print('computed whitespace for', num_processed, ', excluded', num_excluded)

print('Total pictures processed:', num_processed, ', num excluded:', num_excluded)
"""