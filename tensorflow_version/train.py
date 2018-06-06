import tensorflow as tf
import numpy as np
import model
import argparse
from os.path import join
import h5py
from Utils import image_processing
import scipy.misc
import random
import os

from Utils.transfer_learning import transfer_learning


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--z_dim', type=int, default=100,
                        help='Noise dimension')

    parser.add_argument('--t_dim', type=int, default=256,
                        help='Text feature dimension')

    parser.add_argument('--batch_size', type=int, default=64,
                        help='Batch Size')

    parser.add_argument('--image_size', type=int, default=64,
                        help='Image Size a, a x a')

    parser.add_argument('--gf_dim', type=int, default=64,
                        help='Number of conv in the first layer gen.')

    parser.add_argument('--df_dim', type=int, default=64,
                        help='Number of conv in the first layer discr.')

    parser.add_argument('--gfc_dim', type=int, default=1024,
                        help='Dimension of gen untis for for fully connected layer 1024')

    parser.add_argument('--caption_vector_length', type=int, default=2400,
                        help='Caption Vector Length')

    parser.add_argument('--learning_rate', type=float, default=0.0002,
                        help='Learning Rate')

    parser.add_argument('--beta1', type=float, default=0.5,
                        help='Momentum for Adam Update')

    parser.add_argument('--gen_updates', type=int, default=10,
                        help='Generator updates per discriminator update')

    parser.add_argument('--epochs', type=int, default=200,
                        help='Max number of epochs')

    parser.add_argument('--resume_model', type=str, default=None,
                        help='Pre-Trained Model Path, to resume from')

    parser.add_argument('--resume_epoch', type=int, default=0,
                        help='Number of epochs already trained')

    parser.add_argument('--image_dir', type=str, default="Data/mscoco_raw/processed",
                        help='Directory of image')

    parser.add_argument('--style_image_dir', type=str, default="Data/style/item_mafia",
                        help='Directory of style images')

    parser.add_argument('--experiment', type=str, default="default",
                        help='Experiment to save to and load captions for')

    parser.add_argument('--transfer', action='store_true',
                        help='does transfer learning')

    parser.add_argument('--split', type=str, default="train",
                        help='use val for validation set, train for train\
                        mostly a debug flag')

    parser.add_argument('--extra_32', action='store_true',
                        help='extra conv layer when the image is at size 32')

    parser.add_argument('--extra_64', action='store_true',
                        help='extra conv layer when the image is at size 64')

    parser.add_argument('--vgg', action='store_true',
                        help='use vgg like layout')

    args = parser.parse_args()
    if args.vgg and (args.extra_32 or args.extra_64):
        raise Exception("Cannot perform both vgg and extra_x mods at the same time")

    model_options = {
        'z_dim': args.z_dim,
        't_dim': args.t_dim,
        'batch_size': args.batch_size,
        'image_size': args.image_size,
        'gf_dim': args.gf_dim,
        'df_dim': args.df_dim,
        'gfc_dim': args.gfc_dim,
        'caption_vector_length': args.caption_vector_length,
        'extra_32': args.extra_32,
        'extra_64': args.extra_64,
        'vgg': args.vgg
    }

    tbdir = "Data/Experiments/{}/".format(args.experiment)
    tbpath = os.path.join(tbdir, "tensorboard")
    if not os.path.isdir(tbdir):
        os.makedirs(tbdir)
    if not os.path.isdir(tbpath):
        os.makedirs(tbpath)
    tbwriter = tf.summary.FileWriter(tbpath)

    gan = model.GAN(model_options)
    input_tensors, variables, loss, outputs, checks = gan.build_model()

    d_optim = tf.train.AdamOptimizer(args.learning_rate, beta1=args.beta1).minimize(loss['d_loss'],
                                                                                    var_list=variables['d_vars'])
    s_optim = tf.train.AdamOptimizer(args.learning_rate, beta1=args.beta1).minimize(loss['s_loss'],
                                                                                    var_list=variables['s_vars'])
    g_optim = tf.train.AdamOptimizer(args.learning_rate, beta1=args.beta1).minimize(loss['g_loss'],
                                                                                    var_list=variables['g_vars'])

    sess = tf.InteractiveSession()
    tf.global_variables_initializer().run()

    checkpointer = tf.train.Saver()
    perm_saver = tf.train.Saver(max_to_keep=None)

    if args.transfer:
        transfer_learning(sess)
    if args.resume_model:
        checkpointer.restore(sess, args.resume_model)

    loaded_data = load_training_data(args.split, args.experiment)

    for i in range(args.resume_epoch, args.epochs + 1):
        batch_no = 0
        gen_images = None
        random.shuffle(loaded_data['image_list'])
        print(loaded_data['data_length'])
        while batch_no * args.batch_size < loaded_data['data_length']:
            real_images, wrong_images, style_images, caption_vectors, z_noise, image_files = \
                get_training_batch(batch_no,
                                   args.batch_size,
                                   args.image_size,
                                   args.z_dim,
                                   args.caption_vector_length,
                                   args.image_dir,
                                   loaded_data,
                                   args.style_image_dir)

            # DISCR UPDATE
            check_ts = [checks['d_loss1'], checks['d_loss2'], checks['d_loss3']]
            _, d_loss, gen, d1, d2, d3 = sess.run([d_optim, loss['d_loss'], outputs['generator']] + check_ts,
                                                  feed_dict={
                                                      input_tensors['t_real_image']: real_images,
                                                      input_tensors['t_wrong_image']: wrong_images,
                                                      input_tensors['t_real_caption']: caption_vectors,
                                                      input_tensors['t_z']: z_noise,
                                                  })

            print("d1", d1)
            print("d2", d2)
            print("d3", d3)
            print("D", d_loss)

            # STYLE UPDATE
            check_ts = [checks['s_loss_real'], checks['s_loss_wrong'], checks['s_loss_fake']]
            _, s_loss, gen, s1, s2, s3 = sess.run([s_optim, loss['s_loss'], outputs['generator']] + check_ts,
                                                  feed_dict={
                                                      input_tensors['t_real_image']: real_images,
                                                      input_tensors['t_style_image']: style_images,
                                                      input_tensors['t_wrong_image']: wrong_images,
                                                      input_tensors['t_real_caption']: caption_vectors,
                                                      input_tensors['t_z']: z_noise,
                                                  })

            print("s1", s1)
            print("s2", s2)
            print("s3", s3)
            print("S", s_loss)

            g_loss, g1, g2 = None, None, None
            for _ in range(args.gen_updates):
                # GEN UPDATE
                check_ts = [checks['g_loss_content'], checks['g_loss_style']]

                _, g_loss, gen_images, g1, g2 = sess.run([g_optim, loss['g_loss'], outputs['generator']] + check_ts,
                                                         feed_dict={
                                                             input_tensors['t_real_image']: real_images,
                                                             input_tensors['t_wrong_image']: wrong_images,
                                                             input_tensors['t_real_caption']: caption_vectors,
                                                             input_tensors['t_z']: z_noise,
                                                         })

            summary = tf.Summary(value=[tf.Summary.Value(tag="d_loss", simple_value=d_loss),
                                        tf.Summary.Value(tag="d_loss1", simple_value=d1),
                                        tf.Summary.Value(tag="d_loss2", simple_value=d2),
                                        tf.Summary.Value(tag="d_loss3", simple_value=d3),
                                        tf.Summary.Value(tag="s_loss", simple_value=s_loss),
                                        tf.Summary.Value(tag="s_loss_real", simple_value=s1),
                                        tf.Summary.Value(tag="s_loss_wrong", simple_value=s2),
                                        tf.Summary.Value(tag="s_loss_fake", simple_value=s3),
                                        tf.Summary.Value(tag="g_loss", simple_value=g_loss),
                                        tf.Summary.Value(tag="g_loss_content", simple_value=g1),
                                        tf.Summary.Value(tag="g_loss_style", simple_value=g2)])
            global_step = i * loaded_data['data_length'] / args.batch_size + batch_no
            tbwriter.add_summary(summary, global_step)
            print("Epoch", i, "LOSSES", d_loss, g_loss, batch_no, i, loaded_data['data_length'] / args.batch_size)
            batch_no += 1
            checkpointer.save(sess, "Data/Experiments/{}/model/checkpoint.ckpt".format(args.experiment), global_step=i)
        if i > 0 and (i % 100) == 0:
            print("Saving Images, Model")
            save_for_vis(args.experiment, gen_images)
            perm_saver.save(sess, "Data/Experiments/{}/model/after_{}_epochs.ckpt".format(args.experiment, i))


def load_training_data(split, experiment):
    h = h5py.File(os.path.join("Data", "Experiments", experiment, '{}_captions.hdf5'.format(split)))
    captions = {}
    image_list = []
    for class_name in list(h.keys()):
        for ds in h[class_name].items():
            captions[ds[0]] = np.array(ds[1])
            image_list += [(class_name, ds[0])]

    random.shuffle(image_list)

    h.close()

    return {
        'image_list': image_list,
        'captions': captions,
        'data_length': len(image_list),
    }


def save_for_vis(experiment, generated_images):
    train_samples_path = "Data/Experiments/{}/train_samples".format(experiment)
    if not os.path.isdir(train_samples_path):
        os.makedirs(train_samples_path)

    for i in range(0, generated_images.shape[0]):
        fake_images_255 = (generated_images[i, :, :, :])
        scipy.misc.imsave(join(train_samples_path, 'fake_image_{}.jpg'.format(i)), fake_images_255)


def get_training_batch(batch_no, batch_size, image_size,
                       z_dim, caption_vector_length, image_dir,
                       loaded_data, style_img_dir):
    real_images = np.zeros((batch_size, 64, 64, 3))
    wrong_images = np.zeros((batch_size, 64, 64, 3))
    style_images = np.zeros((batch_size, 64, 64, 3))
    captions = np.zeros((batch_size, caption_vector_length))

    cnt = 0
    image_files = []
    for i in range(batch_no * batch_size, batch_no * batch_size + batch_size):
        idx = i % len(loaded_data['image_list'])
        class_name, image_file_name = loaded_data['image_list'][idx]
        processed_img_dir = os.path.join(image_dir, class_name)
        image_file = join(processed_img_dir, image_file_name)
        image_array = image_processing.load_image_array(image_file, image_size)
        real_images[cnt, :, :, :] = image_array

        # Improve this selection of wrong image
        wrong_image_id = random.randint(0, len(loaded_data['image_list']) - 1)
        wrong_class_name, wrong_image_file_name = loaded_data['image_list'][wrong_image_id]
        wrong_processed_img_dir = os.path.join(image_dir, wrong_class_name)
        wrong_image_file = join(wrong_processed_img_dir, wrong_image_file_name)
        wrong_image_array = image_processing.load_image_array(wrong_image_file, image_size)
        wrong_images[cnt, :, :, :] = wrong_image_array

        random_caption = random.randint(0, 4)
        captions[cnt, :] = loaded_data['captions'][image_file_name][random_caption][
                           0:caption_vector_length]
        image_files.append(image_file)
        cnt += 1

    style_candidates = os.listdir(style_img_dir)
    style_image_idx = np.random.choice(range(len(style_candidates)), size=batch_size, replace=False)
    style_image_files = [style_candidates[i] for i in style_image_idx]
    for idx in range(len(style_image_files)):
        image_file = os.path.join(style_img_dir, style_image_files[idx])
        image_array = image_processing.load_image_array(image_file, image_size)
        style_images[idx, :, :, :] = image_array

    z_noise = np.random.uniform(-1, 1, [batch_size, z_dim])
    return real_images, wrong_images, style_images, captions, z_noise, image_files


if __name__ == '__main__':
    main()
