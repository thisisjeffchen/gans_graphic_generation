import tensorflow as tf
import numpy as np
import model
import argparse
from os.path import join
import h5py
import scipy.misc
import os
import shutil


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('--z_dim', type=int, default=100,
                        help='Noise Dimension')

    parser.add_argument('--t_dim', type=int, default=256,
                        help='Text feature dimension')

    parser.add_argument('--image_size', type=int, default=64,
                        help='Image Size')        

    parser.add_argument('--gf_dim', type=int, default=64,
                        help='Number of conv in the first layer gen.')

    parser.add_argument('--df_dim', type=int, default=64,
                        help='Number of conv in the first layer discr.')

    parser.add_argument('--gfc_dim', type=int, default=1024,
                        help='Dimension of gen untis for for fully connected layer 1024')

    parser.add_argument('--caption_vector_length', type=int, default=2400,
                        help='Caption Vector Length')

    parser.add_argument('--n_images', type=int, default=4,
                        help='Number of Images per Caption')

    parser.add_argument('--split', type=str, default='gen',
                        help='train/val/test/gen')

    parser.add_argument('--experiment', type=str, default="default",
                        help='Experiment of dataset')

    parser.add_argument('--epoch', type=int, default=None,
                        help='Epoch of the trained model to load. Defaults to latest checkpoint')

    args = parser.parse_args()
    model_options = {
        'z_dim': args.z_dim,
        't_dim': args.t_dim,
        'batch_size': args.n_images,
        'image_size': args.image_size,
        'gf_dim': args.gf_dim,
        'df_dim': args.df_dim,
        'gfc_dim': args.gfc_dim,
        'caption_vector_length': args.caption_vector_length
    }

    data_dir = os.path.join("Data", "Experiments", "{}".format(args.experiment))
    caption_thought_vectors = os.path.join(data_dir,  '{}_captions.hdf5'.format(args.split))
    save_dir = os.path.join(data_dir, "{}_samples".format(args.split))


    model_path = os.path.join(data_dir, "model")
    checkpoint = tf.train.latest_checkpoint(model_path)
    if args.epoch is not None:
        checkpoint = os.path.join(model_path, "after_{}_epochs.ckpt".format(args.epoch))
        save_dir = os.path.join(save_dir, "epoch_{}".format(args.epoch))
    else:
        save_dir = os.path.join(save_dir, "latest".format(args.epoch))

    gan = model.GAN(model_options)
    _, _, _, _, _ = gan.build_model()
    sess = tf.InteractiveSession()
    saver = tf.train.Saver()
    saver.restore(sess, checkpoint)

    input_tensors, outputs = gan.build_generator()

    h = h5py.File(caption_thought_vectors)
    caption_vectors = {}
    if args.split == 'gen':
        caption_vectors["generated"] = np.array(h['vectors'])
    else:
        class_name = list(h.keys())[0]
        for img_file, vector in h[class_name].items():
            img_id = os.path.splitext(img_file)[0]
            caption_vectors[img_id] = np.array(vector)

    generated_images = {}
    for img_id, vectors in caption_vectors.items():
        caption_image_dic = {}
        for cn, caption_vector in enumerate(vectors):
            caption_images = []
            z_noise = 1 * np.random.uniform(-1, 1, [args.n_images, args.z_dim])
            caption = [caption_vector[0:args.caption_vector_length]] * args.n_images

            [gen_image] = sess.run([outputs['generator']],
                                   feed_dict={
                                       input_tensors['t_real_caption']: caption,
                                       input_tensors['t_z']: z_noise,
                                   })

            caption_images = [gen_image[i, :, :, :] for i in range(0, args.n_images)]
            caption_image_dic[cn] = caption_images
            print("Generated {} images for {}".format(cn, img_id))
        generated_images[img_id] = caption_image_dic

    if os.path.isdir(save_dir):
        shutil.rmtree(save_dir)

    os.makedirs(save_dir, exist_ok=True)

    for img_id, caption_image_dic in generated_images.items():
        for cn, images in caption_image_dic.items():
            for i, im in enumerate(images):             
                scipy.misc.imsave( 
                    join(save_dir, '{}_{}_image_{}_{}.jpg'.format( 
                    img_id, args.image_size, cn, chr (ord('A') + i))), im)



if __name__ == '__main__':
    main()
