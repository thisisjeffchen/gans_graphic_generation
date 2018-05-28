#!/bin/bash

#TODO: add to this if it doesn't work for you!
sudo apt-get install -y emacs  python-pip python-tk python3-pip python-opencv python3-tk

pip install cython nltk tensorflow-gpu matplotlib numpy requests opencv-python pycocotools 
pip install h5py nltk numpy scipy scikit_image tensorflow-gpu Theano


pip3 install cython nltk  matplotlib numpy requests opencv-python pycocotools
pip3 install h5py nltk numpy scipy scikit_image  Theano tensorflow-gpu


cd cocoapi-master/PythonAPI
sudo make
sudo make install
sudo python setup.py install

cd ../../
