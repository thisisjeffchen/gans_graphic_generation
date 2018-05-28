# Execute this file in the mscoco_raw directory
import os
import json
import cv2

import pycocotools.coco as coco

MSCOCO_RAW_DIR = "Data/mscoco_raw/"
PROCESSED_DIR = os.path.join(MSCOCO_RAW_DIR, "processed")

mycoco = coco.COCO('Data/mscoco_raw/annotations/instances_train2017.json')

annotations = {}


if not os.path.isdir(PROCESSED_DIR):
  print ("Processed dir not found... creating ...")
  os.mkdir(PROCESSED_DIR)

with open('Data/mscoco_raw/annotations/instances_val2017.json') as f:
  data = json.load(f)
  for anno in data['annotations']:
    annotations[int(anno['image_id'])] = anno

with open('Data/mscoco_raw/annotations/instances_train2017.json') as f:
  data = json.load(f)
  for anno in data['annotations']:
    annotations[int(anno['image_id'])] = anno

print('Num annotations loaded:', len(annotations))

pics_processed = 0
pics_skipped = 0
for filename in os.listdir('Data/mscoco_raw/train2017'):
  if filename.endswith('.jpg'):
    img = cv2.imread('Data/mscoco_raw/train2017' + '/' + filename)
    img_id = int(filename[0:-4])
    if img_id not in annotations:
      print('Missing annotation for img:', img_id)
      pics_skipped += 1
      continue
    img[mycoco.annToMask(annotations[img_id]) == 0] = [255, 255, 255]
    cv2.imwrite('Data/mscoco_raw/processed' + '/' + filename, img)
    pics_processed += 1
    if pics_processed % 1000 == 0:
      print('finished: ', pics_processed)

print('Total pictures processed:', pics_processed)
print('Total pictures skipped:', pics_skipped)
