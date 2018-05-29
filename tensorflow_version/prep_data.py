from __future__ import print_function
import os
from collections import defaultdict
import argparse
import skipthoughts
import h5py

from pycocotools.coco import COCO

NUM_BATCHES = 32

def prep_data(data_dir, split, target_dir, experiment, category):
    ANN_PATH = os.path.join(data_dir, "annotations/instances_" + split + "2017.json")
    CAP_PATH = os.path.join(data_dir, "annotations/captions_" + split + "2017.json")
    target_dir_experiment = os.path.join(target_dir, experiment)
    TARGET_DIR_CAPS = os.path.join(target_dir_experiment, "captions")

    if not os.path.isdir(target_dir):
        os.mkdir(target_dir)
    if not os.path.isdir(target_dir_experiment):
        os.mkdir(target_dir_experiment)
    if not os.path.isdir(TARGET_DIR_CAPS):
        os.mkdir(TARGET_DIR_CAPS)

    cocoAnn = COCO(ANN_PATH)
    cocoCap = COCO(CAP_PATH)

    catId = cocoAnn.getCatIds([category])[0]
    class_name = "class_{}".format(str(catId).zfill(5))

    image_captions = defaultdict(list)
    class_dir = os.path.join(TARGET_DIR_CAPS,  class_name)
    if not os.path.isdir(class_dir):
        os.mkdir(class_dir)

    skipped_imgs = 0
    imgIds = cocoAnn.getImgIds(catIds=catId)

    for progress, i in enumerate(imgIds):
        if progress % 1000 == 0:
            print("Processsed {} out of {} for class {}".format(progress, len(imgIds), catId))

        img_file = os.path.join(data_dir, "processed", class_name, "{}.jpg".format(str(i).zfill(12)))
        if not os.path.isfile(img_file):
            print("Image {} not found".format(img_file))
            skipped_imgs += 1
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
    print("Skipped {} missing images for {}".format(skipped_imgs, category))
    print("all captions present for " + category)
    return image_captions, class_name

def save_caption_vectors(image_captions, target_dir, split, experiment, class_name):
    import time
    print("number of images: ",  len(image_captions))


    img_batches = [[] for i in range(NUM_BATCHES)]
    caption_batches = [[] for i in range(NUM_BATCHES)]
    counter = 0
    for img, captions in image_captions.items():
        counter = counter % NUM_BATCHES
        img_batches[counter].append(img)
        caption_batches[counter] += captions
        counter +=1
    print("batched")

    h = h5py.File(os.path.join(target_dir, experiment, '{}_captions.hdf5'.format(split)))
    group = h.create_group(class_name)
    model = skipthoughts.load_model()
    for i in range(NUM_BATCHES):
        st = time.time()
        imgs = img_batches[i]
        captions = caption_batches[i]
        encoded_captions = skipthoughts.encode(model, captions)

        cstart = 0
        for img in imgs:
            num_caps = len(image_captions[img])
            print(cstart, num_caps, len(encoded_captions))
            group.create_dataset(img, data=encoded_captions[cstart:cstart+num_caps])
            cstart += num_caps

        print("Batch {} of {} Done".format(i + 1, NUM_BATCHES))
        print("Seconds", time.time() - st)

    h.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--split', type=str, default='train',
                        help='train/val')
    parser.add_argument('--data_dir', type=str, default='Data/mscoco_raw/',
                        help='Data directory')
    parser.add_argument('--experiment', type=str, default="default",
                        help='Experiment of dataset')
    parser.add_argument('--cat', type=str, default="cake",
                        help='The category')


    args = parser.parse_args()

    image_captions, class_name = prep_data(args.data_dir, args.split, "Data/Experiments/", args.experiment, args.cat)
    save_caption_vectors(image_captions, "Data/Experiments/", args.split, args.experiment, class_name)



if __name__ == '__main__':
    main()
