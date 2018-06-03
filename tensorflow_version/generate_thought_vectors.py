import os
import argparse
import skipthoughts
import h5py
from shutil import copyfile


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--experiment', type=str, default="default",
                        help='Experiment of dataset')

    args = parser.parse_args()

    data_dir = "Data/Experiments/{}".format(args.experiment)
    if not os.path.isdir(data_dir):
        os.mkdir(data_dir)

    #copy over caption file into experiment directory
    h = h5py.File(os.path.join("Data", "Experiments", args.experiment, 'train_captions.hdf5'))
    class_names = list(h.keys())
    h.close ()

    out_caption_file = os.path.join(data_dir, "gen_captions.hdf5")

    captions = []
    for class_name in class_names:
        filename = class_name + ".txt"
        src = os.path.join ("DefaultCaptions/", filename)
        dst = os.path.join (data_dir, filename)
        copyfile(src, dst)
        in_caption_file = os.path.join(data_dir, filename)

        with open(in_caption_file) as f:
            captions_text = f.read().split('\n')
            captions += [cap for cap in captions_text if len(cap) > 0]

    print(captions)
    model = skipthoughts.load_model()
    caption_vectors = skipthoughts.encode(model, captions)

    if os.path.isfile(out_caption_file):
        os.remove(out_caption_file)
    h = h5py.File(out_caption_file)
    h.create_dataset('vectors', data=caption_vectors)
    h.close()


if __name__ == '__main__':
    main()
