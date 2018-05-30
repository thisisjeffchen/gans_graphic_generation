import os
import json
import cv2
import numpy as np
import argparse


from pycocotools.coco import COCO

def main ():
    parser = argparse.ArgumentParser()
    parser.add_argument ('--debug', action='store_true',
        help= 'Debug mode works on val dataset')
    parser.add_argument ('--crop', action='store_true', 
        help= 'performs cropping and centering')


    MSCOCO_RAW_DIR = "Data/mscoco_raw/"

    args = parser.parse_args ()
    CROP = args.crop
    DEBUG = args.debug
    
    
    if CROP:
        PROCESSED_DIR = os.path.join(MSCOCO_RAW_DIR, "processed_segmented_cropped")
        DISREGARDED_DIR = os.path.join(MSCOCO_RAW_DIR, "disregarded_segmented_cropped")
    else:
        PROCESSED_DIR = os.path.join(MSCOCO_RAW_DIR, "processed_segmented")
        DISREGARDED_DIR = os.path.join(MSCOCO_RAW_DIR, "disregarded_segmented")

    
    THRESHOLD = 0.07
    WHITE=[255,255,255]
    
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
            if img is None:
                print('Missing img:', imgId)
                continue
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
    
                if CROP:
                    # compute crop dimensions for subject
                    rows = np.any(img-255, axis=1)
                    cols = np.any(img-255, axis=0)
                    rmin, rmax = np.where(rows)[0][[0, -1]]
                    cmin, cmax = np.where(cols)[0][[0, -1]]
                    size = max(rmax - rmin, cmax - cmin)
                    vertical_len = (size - (rmax-rmin))
                    horizontal_len = (size - (cmax-cmin))
                    vertical_pad = int (vertical_len / 2)
                    vertical_add = vertical_len % 2
                    horizontal_pad = int(horizontal_len / 2)
                    horizontal_add = horizontal_len % 2
            
                    img = cv2.copyMakeBorder(
                        img[rmin:rmax, cmin:cmax],
                        vertical_pad + vertical_add, vertical_pad, 
                        horizontal_pad + horizontal_add, horizontal_pad,
                        cv2.BORDER_CONSTANT, value=WHITE)
    
                if percent_subject >= THRESHOLD:
                     
                    cv2.imwrite(os.path.join(class_dir_processed, filename), img)
                    pics_processed += 1
                else:
                    cv2.imwrite(os.path.join(class_dir_disregarded, filename), img)
                    pics_disregarded += 1
    
    print('Total pictures processed:', pics_processed)
    print('Total pictures disregarded due to size:', pics_disregarded)
    print('Total pictures skipped due to no annotaion:', pics_skipped)

if __name__ == '__main__':
    main()
