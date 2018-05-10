#!/bin/bash

#TODO: add to this if it doesn't work for you!
sudo apt-get install -y emacs  python-pip python-tk

sudo pip install cython nltk tensorflow-gpu matplotlib numpy



cd cocoapi-master/PythonAPI
make

cd ../../
