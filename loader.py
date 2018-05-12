import sys
import os
sys.path.append (os.path.abspath("cocoapi-master/PythonAPI"))
import pdb
                 
from pycocotools.coco import COCO


DATASET = "val"

def main ():
    #print_info ()
    prep_data ()


def prep_data ():

    DATA_PATH = "tensorflow_version/Data/mscoco_raw/"
    ANN_PATH = DATA_PATH + "annotations/instances_" + DATASET + "2017.json"
    CAP_PATH = DATA_PATH + "annotations/captions_" + DATASET + "2017.json"

    cocoAnn = COCO (ANN_PATH)
    cocoCap = COCO (CAP_PATH)

    catNames = ["person", "car", "airplane", "boat", "bus", "horse", "elephant",
            "motorcycle", "tv", "refrigerator", "bear"]

    catIds = cocoAnn.getCatIds (catNames)
    assert (len(catIds) == len (catNames))

    for idx, catId in enumerate (catIds):
        imgIds = coco.getImgIds (catIds = catId)
        annIds = cocoCap.getAnnIds (imgIds = imgIds)
        anns = cocoCap.loadAnns (annIds)

        assert (len ())
        print "all captions present for " + catNames[idx]
        #TODO: save this



def print_info ():
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

    
if __name__ == '__main__':
    main()