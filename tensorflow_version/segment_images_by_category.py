import os
import json
import cv2
import numpy as np

from pycocotools.coco import COCO

DEBUG = False

MSCOCO_RAW_DIR = "Data/mscoco_raw/"
PROCESSED_DIR = os.path.join(MSCOCO_RAW_DIR, "processed")
DISREGARDED_DIR = os.path.join(MSCOCO_RAW_DIR, "disregarded")
THRESHOLD = 0.07

if DEBUG:
    TRAIN_DIR = "Data/mscoco_raw/val2017/"
    coco = COCO('Data/mscoco_raw/annotations/instances_val2017.json')
else:
    TRAIN_DIR = "Data/mscoco_raw/train2017/"
    coco = COCO('Data/mscoco_raw/annotations/instances_train2017.json')

if not os.path.isdir(PROCESSED_DIR):
    print("Processed directory not found... creating ...")
    os.mkdir(PROCESSED_DIR)

if not os.path.isdir(DISREGARDED_DIR):
    print("Disregarded directory not found... creating ...")
    os.mkdir(DISREGARDED_DIR)

cats = coco.getCatIds()
cat_names = coco.loadCats(cats)

pics_processed = 0
cat_processed = 0

pics_skipped = 0
pics_disregarded = 0

for i, cat in enumerate(cats):
    # create directory
    class_dir_processed = os.path.join(PROCESSED_DIR, "class_{}".format(str(cat).zfill(5)))
    class_dir_disregarded = os.path.join(DISREGARDED_DIR, "class_{}".format(str(cat).zfill(5)))

    for d in [class_dir_processed, class_dir_disregarded]:
        if not os.path.isdir(d):
            os.mkdir(d)

    print(str(cat))
    print(cat_names[i])
    imgIds = coco.getImgIds(catIds=cat)
    for imgId in imgIds:
        filename = str(imgId).zfill(12) + '.jpg'
        img = cv2.imread(TRAIN_DIR + filename)
        # We only take the 1st annotation on the image
        annIds = coco.getAnnIds(imgIds=imgId, catIds=cat)
        if (len(annIds) == 0):
            print('Missing annotation for img:', imgId)
            pics_skipped += 1
        else:
            # Choose the first annotation for the category only (hope it's the best one)
            ann = coco.loadAnns(annIds)[0]
            img[coco.annToMask(ann) == 0] = [255, 255, 255]

            num_white = np.sum(img == 255)
            num_total = img.shape[0] * img.shape[1] * img.shape[2]
            percent_subject = 1.0 - (num_white * 1.0 / num_total)

            # compute crop dimensions for subject
            rows = np.any(img-255, axis=1)
            cols = np.any(img-255, axis=0)
            rmin, rmax = np.where(rows)[0][[0, -1]]
            cmin, cmax = np.where(cols)[0][[0, -1]]

            if percent_subject >= THRESHOLD:
                cv2.imwrite(os.path.join(class_dir_processed, filename), img[rmin:rmax, cmin:cmax])
                pics_processed += 1
            else:
                cv2.imwrite(os.path.join(class_dir_disregarded, filename), img[rmin:rmax, cmin:cmax])
                pics_disregarded += 1

print('Total pictures processed:', pics_processed)
print('Total pictures disregarded due to size:', pics_disregarded)
print('Total pictures skipped due to no annotaion:', pics_skipped)
