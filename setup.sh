#!/bin/bash

#TODO: add to this if it doesn't work for you!
sudo apt-get install -y emacs  python-pip python-tk python3-pip python-opencv python3-tk

pip install nltk tensorflow-gpu matplotlib numpy requests opencv-python pycocotools future
pip install h5py nltk numpy scipy scikit_image tensorflow-gpu Theano cython


pip3 install  nltk  matplotlib numpy requests opencv-python pycocotools future
pip3 install h5py nltk numpy scipy scikit_image  Theano tensorflow-gpu cython


cd cocoapi-master/PythonAPI
sudo make
sudo make install
sudo python3 setup.py install

cd ../../

ln -s  /home/shared/mscoco_raw tensorflow_version/Data/mscoco_raw
