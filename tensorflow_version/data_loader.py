import json
import os
from os.path import join, isfile
import re
import numpy as np
import pickle
import argparse
import skipthoughts
import h5py


# DID NOT TRAIN IT ON MS COCO YET
def save_caption_vectors_ms_coco(data_dir, split, batch_size):
    meta_data = {}
    ic_file = join(data_dir, 'annotations/captions_{}2017.json'.format(split))
    with open(ic_file) as f:
        ic_data = json.loads(f.read())

    meta_data['data_length'] = len(ic_data['annotations'])
    with open(join(data_dir, 'meta_{}.pkl'.format(split)), 'wb') as f:
        pickle.dump(meta_data, f)

    model = skipthoughts.load_model()
    batch_no = 0
    print("Total Batches", len(ic_data['annotations']) / batch_size)

    while batch_no * batch_size < len(ic_data['annotations']):
        captions = []
        image_ids = []
        idx = batch_no
        for i in range(batch_no * batch_size, (batch_no + 1) * batch_size):
            idx = i % len(ic_data['annotations'])
            captions.append(ic_data['annotations'][idx]['caption'])
            image_ids.append(ic_data['annotations'][idx]['image_id'])

        print(captions)
        print(image_ids)
        # Thought Vectors
        tv_batch = skipthoughts.encode(model, captions)
        h5f_tv_batch = h5py.File(join(data_dir, 'tvs/' + split + '_tvs_' + str(batch_no)), 'w')
        h5f_tv_batch.create_dataset('tv', data=tv_batch)
        h5f_tv_batch.close()

        h5f_tv_batch_image_ids = h5py.File(join(data_dir, 'tvs/' + split + '_tv_image_id_' + str(batch_no)), 'w')
        h5f_tv_batch_image_ids.create_dataset('tv', data=image_ids)
        h5f_tv_batch_image_ids.close()

        print("Batches Done", batch_no, len(ic_data['annotations']) / batch_size)
        batch_no += 1


def save_caption_vectors(data_dir, args):
    import time

    img_dir = join(data_dir, args.data_set + '/jpg')
    image_files = [f for f in os.listdir(img_dir) if 'jpg' in f]
    print(image_files[300:400])
    print("number of images: " + str(len(image_files)))
    image_captions = {img_file: [] for img_file in image_files}

    caption_dir = join(data_dir, args.data_set + '/text_c10')
    class_dirs = []
    for d in os.listdir(caption_dir):
        class_dirs.append(join(caption_dir, d))

    for class_dir in class_dirs:
        caption_files = [f for f in os.listdir(class_dir) if 'txt' in f]
        for cap_file in caption_files:
            with open(join(class_dir, cap_file)) as f:
                captions = f.read().split('\n')
            img_file = cap_file.split('.')[0] + ".jpg"
            # 5 captions per image
            captions = [cap.strip() for cap in captions if len(cap.strip()) > 0][0:5]
            image_captions[img_file] += captions

    print(len(image_captions))

    img_batches = [[] for i in range(args.batch_size)]
    caption_batches = [[] for i in range(args.batch_size)]
    counter = 0
    for img, captions in image_captions.items():
        counter = counter % args.batch_size
        img_batches[counter].append(img)
        caption_batches[counter] += captions
        counter +=1
    print("batched")

    model = skipthoughts.load_model()
    h = h5py.File(join(data_dir, args.data_set + '_tv.hdf5'))
    for i in range(args.batch_size):
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

        print("Batch {} of {} Done".format(i, args.batch_size))
        print("Seconds", time.time() - st)

    h.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--split', type=str, default='train',
                        help='train/val')
    parser.add_argument('--data_dir', type=str, default='Data',
                        help='Data directory')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='a Size')
    parser.add_argument('--data_set', type=str, default='flowers',
                        help='Data Set : Flowers, MS-COCO')
    args = parser.parse_args()

    save_caption_vectors(args.data_dir, args)
    

if __name__ == '__main__':
    main()
