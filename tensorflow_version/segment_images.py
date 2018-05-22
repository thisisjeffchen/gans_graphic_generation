# Execute this file in the mscoco_raw directory
import os
import json
import cv2

import pycocotools.coco as coco

mycoco = coco.COCO('annotations/instances_val2017.json')

annotations = {}

with open('annotations/instances_val2017.json') as f:
  data = json.load(f)
  for anno in data['annotations']:
    annotations[int(anno['image_id'])] = anno

print('Num annotations loaded:', len(annotations))

pics_processed = 0
pics_skipped = 0
for filename in os.listdir('./train2017'):
  if filename.endswith('.jpg'):
    img = cv2.imread('./train2017' + '/' + filename)
    img_id = int(filename[0:-4])
    if img_id not in annotations:
      print('Missing annotation for img:', img_id)
      pics_skipped += 1
      continue
    img[mycoco.annToMask(annotations[img_id]) == 0] = [255, 255, 255]
    cv2.imwrite('./processed' + '/' + filename, img)
    pics_processed += 1

print('Total pictures processed:', pics_processed)
print('Total pictures skipped:', pics_skipped)
