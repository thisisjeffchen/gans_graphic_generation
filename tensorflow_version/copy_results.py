import argparse
import os
import glob
import shutil
import errno

GEN_SAMPLES_DIR = "gen_samples"
STYLE_SAMPLES_DIR = "styled_gen_samples"

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--experiment', type=str, default="default",
                        help='Experiment of dataset')

    args = parser.parse_args()
    
    dest_path = os.path.join ("../results/", args.experiment)
    from_path = "Data/Experiments/{}/".format(args.experiment)
        
    try:
        os.makedirs(dest_path)
    except OSError as exception:
        if exception.errno != errno.EEXIST:
            raise

    #Copy over images
    shutil.rmtree(os.path.join(dest_path, GEN_SAMPLES_DIR))
    shutil.copytree (os.path.join (from_path, GEN_SAMPLES_DIR),
                     os.path.join (dest_path, GEN_SAMPLES_DIR))

    #Copy over styled images
    complete_style_path = os.path.join(from_path, STYLE_SAMPLES_DIR)
    if os.path.exists(complete_style_path):
        shutil.rmtree(os.path.join(dest_path, STYLE_SAMPLES_DIR))
        shutil.copytree(complete_style_path,
                        os.path.join(dest_path, STYLE_SAMPLES_DIR))

    for filename in glob.glob(os.path.join(from_path, '*.txt')):
        shutil.copy(filename, dest_path)




if __name__ == '__main__':
        main()
        
