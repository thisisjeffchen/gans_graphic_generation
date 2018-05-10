import sys
import os
sys.path.append (os.path.abspath("cocoapi-master/PythonAPI"))
                 
from pycocotools.coco import COCO

DATA_PATH = "/home/shared/cs231n_data"
TRAIN_INSTANCES_ANN = "annotations/instances_train2017.json"
VAL_INSTANCES_ANN = "annotations/instances_val2017.json"

ds = {}

ds["train"] = COCO (DATA_PATH + "/" + TRAIN_INSTANCES_ANN)
ds["val"] = COCO (DATA_PATH + "/" + VAL_INSTANCES_ANN)


for k, coco in ds.iteritems ():
    if (k == "train"):
        print "***Training Set:***"
    elif (k == "val"):
        print "***Validation Set:***"

    catIdsAll = coco.getCatIds()
    cats = coco.loadCats(catIdsAll)
    nms=[cat['name'] for cat in cats]
#    print('COCO categories: \n{}\n'.format(' '.join(nms)))

    assert len (cats) == len (nms)
    assert len (catIdsAll) == len (cats)
    for idx in range (len (catIdsAll)):
        imgIds = coco.getImgIds (catIds = catIdsAll[idx])
        num_imgs = len(imgIds)

        print nms[idx] + " " + str(num_imgs)
        
    
    nms = set([cat['supercategory'] for cat in cats])
    print('COCO supercategories: \n{}'.format(' '.join(nms)))

    



                 
