from __future__ import print_function
import sys
import os
from shutil import copyfile

sys.path.append(os.path.abspath("cocoapi-master/PythonAPI"))
import pdb

from pycocotools.coco import COCO

DATASET = "train"


def main():
    # print_info ()
    prep_data()


def prep_data():
    DATA_PATH = "tensorflow_version/Data/mscoco_raw/"
    ANN_PATH = DATA_PATH + "annotations/instances_" + DATASET + "2017.json"
    CAP_PATH = DATA_PATH + "annotations/captions_" + DATASET + "2017.json"
    TARGET_DIR = "tensorflow_version/Data/mscoco/"
    TARGET_DIR_IMAGES = TARGET_DIR + "jpg/"
    TARGET_DIR_CAPS = TARGET_DIR + "text_c10/"

    os.mkdir(TARGET_DIR)
    os.mkdir(TARGET_DIR_IMAGES)
    os.mkdir(TARGET_DIR_CAPS)

    cocoAnn = COCO(ANN_PATH)
    cocoCap = COCO(CAP_PATH)


    catNames = ["car", "airplane", "boat", "bus", "horse", "elephant",
             "motorcycle", "tv", "refrigerator", "bear"]

    catIds = cocoAnn.getCatIds(catNames)
    assert (len(catIds) == len(catNames))

    for idx, catId in enumerate(catIds):
        class_dir = "class_" + str(catId).zfill(5)
        os.mkdir(TARGET_DIR_CAPS + class_dir)

        imgIds = cocoAnn.getImgIds(catIds=catId)

        for progress, i in enumerate(imgIds):

            if progress % 1000 == 0:
                print("Processsed {} out of {} for class {}".format(progress, len(imgIds), catId))

            src = DATA_PATH + DATASET + "2017/" + str(i).zfill(12) + ".jpg"
            dest = TARGET_DIR_IMAGES + "image_" + str(i).zfill(12) + ".jpg"
            copyfile(src, dest)

            cap_file_path = TARGET_DIR_CAPS + class_dir + "/image_" + str(i).zfill(12) + ".txt"
            f = open(cap_file_path, "w")
            annIds = cocoCap.getAnnIds(imgIds=i)
            anns = cocoCap.loadAnns(annIds)
            assert (len(anns) >= 5)  # I think the rest of the code depends on 5 or more caps
            for ann in anns:
                if (ann['caption'].split () > 0):
                    f.write (ann['caption'])
                    f.write ('\n')
            f.close ()
        print "all captions present for " + catNames[idx]


def print_info():
    '''Currently not used  '''
    DATA_PATH = "/home/shared/cs231n_data"
    TRAIN_INSTANCES_ANN = "annotations/instances_train2017.json"
    VAL_INSTANCES_ANN = "annotations/instances_val2017.json"

    ds = {}

    ds["train"] = COCO(DATA_PATH + "/" + TRAIN_INSTANCES_ANN)
    ds["val"] = COCO(DATA_PATH + "/" + VAL_INSTANCES_ANN)

    for k, coco in ds.items():
        if (k == "train"):
            print("***Training Set:***")
        elif (k == "val"):
            print("***Validation Set:***")

        catIdsAll = coco.getCatIds()
        cats = coco.loadCats(catIdsAll)
        nms = [cat['name'] for cat in cats]
        #    print('COCO categories: \n{}\n'.format(' '.join(nms)))

        assert len(cats) == len(nms)
        assert len(catIdsAll) == len(cats)
        for idx in range(len(catIdsAll)):
            imgIds = coco.getImgIds(catIds=catIdsAll[idx])
            num_imgs = len(imgIds)

            print(nms[idx] + " " + str(num_imgs))

        nms = set([cat['supercategory'] for cat in cats])
        print('COCO supercategories: \n{}'.format(' '.join(nms)))


if __name__ == '__main__':
    main()
