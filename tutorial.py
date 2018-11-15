import os
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import math


from keras.layers import Input, Reshape, Conv2D, Flatten, Activation
from keras.layers import BatchNormalization, UpSampling2D
from keras.models import Model, Sequential
from keras.layers.core import Dense, Dropout
from keras.layers.advanced_activations import LeakyReLU
from keras.datasets import mnist, cifar10, fashion_mnist
from keras.optimizers import Adam
from keras import initializers

# Let Keras know that we are using tensorflow as our backend engine
os.environ["KERAS_BACKEND"] = "tensorflow"

# To make sure that we can reproduce the experiment and get the same results
np.random.seed(9)

# The dimension of our random noise vector.
random_dim = 100

image_height = 1
image_width = 1
image_depth = 1
image_size = 1

def load_minst_data():
    # load the data
    new_x = []
    (x_train, labels), (_, _) = cifar10.load_data()
    for i in range(len(labels)):
        if labels[i][0] == 0:
            new_x.append(x_train[i])

    x_train = np.array(new_x)

    print(len(x_train))

    #Setup the dimensions of the images
    global image_width
    global image_height
    global image_depth
    global image_size
    image_width = x_train.shape[1]
    image_height = x_train.shape[2]
    if len(x_train.shape) > 3:
        image_depth = x_train.shape[3]
    #Find the dimension of the image flattened out
    image_size = image_height * image_width

    # normalize our inputs to be in the range[-1, 1]
    x_train = (x_train.astype(np.float32) - 127.5)/127.5
    # convert x_train with a shape of (60000, 28, 28) to (60000, 784) so we have
    # 784 columns per row
    return x_train

# You will use the Adam optimizer
def get_optimizer():
    return Adam(lr=0.0002, beta_1=0.5)

def get_generator(optimizer):
    generator = Sequential()
    dropout_prob = 0.4

    generator.add(Dense(4*4*256, input_dim=100))
    generator.add(BatchNormalization(momentum=0.9))
    generator.add(Activation('relu'))
    generator.add(Reshape((4,4,256)))
    generator.add(Dropout(dropout_prob))

    generator.add(UpSampling2D())
    generator.add(Conv2D(128, 5, padding='same'))
    generator.add(BatchNormalization(momentum=0.9))
    generator.add(Activation('relu'))

    generator.add(UpSampling2D())
    generator.add(Conv2D(128, 5, padding='same'))
    generator.add(BatchNormalization(momentum=0.9))
    generator.add(Activation('relu'))

    generator.add(UpSampling2D())
    generator.add(Conv2D(64, 5, padding='same'))
    generator.add(BatchNormalization(momentum=0.9))
    generator.add(Activation('relu'))

    generator.add(Conv2D(32, 5, padding='same'))
    generator.add(BatchNormalization(momentum=0.9))
    generator.add(Activation('relu'))

    generator.add(Conv2D(3, 5, padding='same'))
    generator.add(Activation('sigmoid'))

    generator.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])

    return generator

def get_discriminator(optimizer):
    discriminator = Sequential()
    input_shape = (32, 32, 3)
    dropout_prob = 0.4

    discriminator.add(Conv2D(32, 5, strides=2, input_shape=input_shape, padding='same'))
    discriminator.add(LeakyReLU())

    discriminator.add(Conv2D(64, 5, strides=2, padding='same'))
    discriminator.add(LeakyReLU())
    discriminator.add(Dropout(dropout_prob))

    discriminator.add(Conv2D(128, 5, strides=2, padding='same'))
    discriminator.add(LeakyReLU())
    discriminator.add(Dropout(dropout_prob))

    discriminator.add(Conv2D(256, 5, strides=1, padding='same'))
    discriminator.add(LeakyReLU())
    discriminator.add(Dropout(dropout_prob))

    discriminator.add(Flatten())
    discriminator.add(Dense(1))
    discriminator.add(Activation('sigmoid'))

    discriminator.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    return discriminator

def get_gan_network(discriminator, random_dim, generator, optimizer):
    # We initially set trainable to False since we only want to train either the
    # generator or discriminator at a time
    discriminator.trainable = False
    # gan input (noise) will be 100-dimensional vectors
    gan_input = Input(shape=(random_dim,))
    # the output of the generator (an image)
    x = generator(gan_input)
    # get the output of the discriminator (probability if the image is real or not)
    gan_output = discriminator(x)
    gan = Model(inputs=gan_input, outputs=gan_output)
    gan.compile(loss='binary_crossentropy', optimizer=optimizer)
    return gan

# Create a wall of generated MNIST images
def plot_generated_images(epoch, generator, examples=100, dim=(10, 10), figsize=(10, 10)):
    noise = np.random.normal(0, 1, size=[examples, random_dim])
    generated_images = generator.predict(noise)

    plt.figure(figsize=figsize)
    for i in range(generated_images.shape[0]):
        plt.subplot(dim[0], dim[1], i+1)
        plt.imshow(generated_images[i], interpolation='nearest', cmap='gray')
        plt.axis('off')
    plt.tight_layout()
    plt.savefig('gan_generated_image_epoch_%d.png' % epoch)

def train(epochs=1, batch_size=128):
    # Get the training and testing data
    x_train = load_minst_data()
    # Split the training data into batches of size 128
    batch_count = x_train.shape[0] / batch_size

    # Build our GAN netowrk
    adam = get_optimizer()
    generator = get_generator(adam)
    discriminator = get_discriminator(adam)
    gan = get_gan_network(discriminator, random_dim, generator, adam)

    for e in range(1, epochs+1):
        print('-'*15, 'Epoch %d' % e, '-'*15)
        for _ in tqdm(range(math.ceil(batch_count))):
            # Get a random set of input noise and images
            noise = np.random.normal(-1.0, 1.0, size=[batch_size, random_dim])
            image_batch = x_train[np.random.randint(0, x_train.shape[0], size=batch_size)]

            # Generate fake MNIST images
            generated_images = generator.predict(noise)
            X = np.concatenate([image_batch, generated_images])

            # Labels for generated and real data
            y_dis = np.zeros(2*batch_size)
            # One-sided label smoothing
            y_dis[:batch_size] = 0.9

            # Train discriminator
            discriminator.trainable = True
            discriminator.train_on_batch(X, y_dis)

            # Train generator
            noise = np.random.normal(0, 1, size=[batch_size, random_dim])
            y_gen = np.ones(batch_size)
            discriminator.trainable = False
            gan.train_on_batch(noise, y_gen)

        if e == 1 or e % 20 == 0:
            plot_generated_images(e, generator)
    plot_generated_images(epochs, generator)

if __name__ == '__main__':
    train(400, 128)
