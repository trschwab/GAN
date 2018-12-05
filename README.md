# Generative Adversarial Network (GAN)

We created a GAN that takes in a set of images, trains on them, and is then able to generate an image that fits into the set.
It works with the cifar10 dataset, a collection of 60,000 images from 10 categories that can easily be loaded with keras.
However, we were unable to expand it to work with a collection of any images, such as flowers.
For some reason, every generated images looked the exact same (mode collapse) and nothing like a flower.

## Instructions to Run

The GAN is setup to run with cifar10 properly.
First, ensure you have all dependencies installed. They can all be installed with pip. You will need:

tensorflow,
keras,
matplotlib,
numpy,
tqdm

Next, simply run with python 3:

python gan.py

Note by default it will run for 100 epochs and print images every 5 epochs, so whenever you are satisfied with the output
you can simply ctrl + c to end the program and you'll still have the images it outputs saved from every 5 epochs.
