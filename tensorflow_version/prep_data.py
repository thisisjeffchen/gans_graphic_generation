from __future__ import print_function
import os
from collections import defaultdict
import argparse
import skipthoughts
import h5py

from pycocotools.coco import COCO

CATEGORIES = ["elephant"] 


def prep_data(data_dir, split, target_dir):
    ANN_PATH = os.path.join(data_dir, "annotations/instances_" + split + "2017.json")
    CAP_PATH = os.path.join(data_dir, "annotations/captions_" + split + "2017.json")
    target_dir_split = os.path.join(target_dir, split)
    TARGET_DIR_CAPS = os.path.join(target_dir_split, "captions")

    if not os.path.isdir(target_dir):
        os.mkdir(target_dir)
    if not os.path.isdir(target_dir_split):
        os.mkdir(target_dir_split)
    if not os.path.isdir(TARGET_DIR_CAPS):
        os.mkdir(TARGET_DIR_CAPS)

    cocoAnn = COCO(ANN_PATH)
    cocoCap = COCO(CAP_PATH)

    catIds = cocoAnn.getCatIds(CATEGORIES)
    assert (len(catIds) == len(CATEGORIES))

    image_captions = defaultdict(list)
    for idx, catId in enumerate(catIds):
        class_dir = os.path.join(TARGET_DIR_CAPS,  "class_{}".format(str(catId).zfill(5)))
        if not os.path.isdir(class_dir):
            os.mkdir(class_dir)

        imgIds = cocoAnn.getImgIds(catIds=catId)

        for progress, i in enumerate(imgIds):
            if progress % 1000 == 0:
                print("Processsed {} out of {} for class {}".format(progress, len(imgIds), catId))

            img_file = os.path.join(data_dir, "{}2017".format(split), str(i).zfill(12) + ".jpg")
            if not os.path.isfile(img_file):
                print("Image {} not found".format(img_file))
            else:
                cap_file_path = os.path.join(class_dir,  str(i).zfill(12) + ".txt")
                with open(cap_file_path, "w") as f:
                    annIds = cocoCap.getAnnIds(imgIds=i)
                    for ann in cocoCap.loadAnns(annIds)[0:5]:
                        cap = ann['caption'].strip()
                        if (len(cap) > 0):
                            f.write(cap)
                            f.write('\n')
                            image_captions[str(i).zfill(12)+".jpg"].append(cap)
        print("all captions present for " + CATEGORIES[idx])
    return image_captions

def save_caption_vectors(image_captions, target_dir, split, num_batches, experiment):
    import time
    print("number of images: ",  len(image_captions))


    img_batches = [[] for i in range(num_batches)]
    caption_batches = [[] for i in range(num_batches)]
    counter = 0
    for img, captions in image_captions.items():
        counter = counter % num_batches
        img_batches[counter].append(img)
        caption_batches[counter] += captions
        counter +=1
    print("batched")

    h = h5py.File(os.path.join(target_dir, split, '{}_captions.hdf5'.format(experiment)))
    model = skipthoughts.load_model()
    for i in range(num_batches):
        st = time.time()
        imgs = img_batches[i]
        captions = caption_batches[i]
        encoded_captions = skipthoughts.encode(model, captions)

        cstart = 0
        for img in imgs:
            num_caps = len(image_captions[img])
            print(cstart, num_caps, len(encoded_captions))
            h.create_dataset(img, data=encoded_captions[cstart:cstart+num_caps])
            cstart += num_caps

        print("Batch {} of {} Done".format(i+1, num_batches))
        print("Seconds", time.time() - st)

    h.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--split', type=str, default='train',
                        help='train/val')
    parser.add_argument('--data_dir', type=str, default='Data/mscoco_raw/',
                        help='Data directory')
    parser.add_argument('--target_dir', type=str, default='Data/mscoco/',
                        help='Target directory')
    parser.add_argument('--num_batches', type=int, default=64,
                        help='a Size')
    parser.add_argument('--experiment', type=str, default="default",
                        help='Experiment of dataset')


    args = parser.parse_args()

    image_captions = prep_data(args.data_dir, args.split, args.target_dir)
    save_caption_vectors(image_captions, args.target_dir, args.split, args.num_batches, args.experiment)



if __name__ == '__main__':
    main()
