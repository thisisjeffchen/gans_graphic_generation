# Execute this post-processing script to remove whitespace in the segmented images
import os
import json
import cv2
import numpy as np
from shutil import copyfile


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
      # subject is large enough, so crop and save to new location
      rows = np.any(img-255, axis=1)
      cols = np.any(img-255, axis=0)
      rmin, rmax = np.where(rows)[0][[0, -1]]
      cmin, cmax = np.where(cols)[0][[0, -1]]
      # reload the image because existing is grayscale
      img = cv2.imread(os.path.join(PROCESSED_DIR, filename))
      cv2.imwrite(os.path.join(OUTPUT_DIR, filename), img[rmin:rmax, cmin:cmax])
    else:
        num_excluded += 1

    if num_processed % 1000 == 0:
        print('computed whitespace for', num_processed, ', excluded', num_excluded)

print('Total pictures processed:', num_processed, ', num excluded:', num_excluded)
