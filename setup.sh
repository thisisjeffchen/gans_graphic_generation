#!/bin/bash

#TODO: add to this if it doesn't work for you!
sudo apt-get install -y emacs  python-pip python-tk python3-pip

pip3 install cython nltk tensorflow-gpu matplotlib numpy requests
pip3 install h5py nltk numpy scipy scikit_image tensorflow-gpu Theano


cd cocoapi-master/PythonAPI
make

cd ../../
