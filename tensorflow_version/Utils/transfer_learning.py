import os
import sys
from six.moves import urllib
import tensorflow as tf

MODEL_DIR = '/tmp/pretrained_flower'
DATA_URL = 'https://bitbucket.org/paarth_neekhara/texttomimagemodel/raw/74a4bbaeee26fe31e148a54c4f495694680e2c31/latest_model_flowers_temp.ckpt'


def download_model():
    if not os.path.exists(MODEL_DIR):
        os.makedirs(MODEL_DIR)
    filename = DATA_URL.split('/')[-1]
    filepath = os.path.join(MODEL_DIR, filename)
    if not os.path.exists(filepath):
        def _progress(count, block_size, total_size):
            sys.stdout.write('\r>> Downloading %s %.1f%%' % (
                filename, float(count * block_size) / float(total_size) * 100.0))
            sys.stdout.flush()


        filepath, _ = urllib.request.urlretrieve(DATA_URL, filepath, _progress)
        print()
        statinfo = os.stat(filepath)
        print('Succesfully downloaded', filename, statinfo.st_size, 'bytes.')
    return filepath

def transfer_learning(sess):
    path = download_model()
    with tf.variable_scope("", reuse=True):
        load_vars = {"d_h0_conv/biases": tf.get_variable("discriminator/d_h0_conv/biases"),
                     "d_h0_conv/w": tf.get_variable("discriminator/d_h0_conv/w"),
                     "d_h1_conv/biases": tf.get_variable("discriminator/d_h1_conv/biases"),
                     "d_h1_conv/w": tf.get_variable("discriminator/d_h1_conv/w"),
                     "d_bn1/beta": tf.get_variable("discriminator/d_bn1/beta"),
                     "d_bn1/gamma": tf.get_variable("discriminator/d_bn1/gamma"),
                     "g_bn0/beta": tf.get_variable("g_bn0/beta"),
                     "g_bn0/gamma": tf.get_variable("g_bn0/gamma"),
                     "g_bn1/beta": tf.get_variable("g_bn1/beta"),
                     "g_bn1/gamma": tf.get_variable("g_bn1/gamma"),
                     "g_embedding/Matrix": tf.get_variable("g_embedding/Matrix"),
                     "g_embedding/bias": tf.get_variable("g_embedding/bias"),
                     "g_h0_lin/Matrix": tf.get_variable("g_h0_lin/Matrix"),
                     "g_h0_lin/bias": tf.get_variable("g_h0_lin/bias"),
                     "g_h1/biases": tf.get_variable("g_h1/biases"),
                     "g_h1/w": tf.get_variable("g_h1/w")
                     }
        saver = tf.train.Saver(load_vars)
        saver.restore(sess, path)