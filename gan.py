import tensorflow as tf
from keras.datasets import cifar10, mnist
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import os
#from scipy import misc

### This code was used when we tried to get it to work on images of flowers
# path = './flowers'
# i = 0
## First reduce the size of all the images
# for image_path in os.listdir(path):
#     input_path = os.path.join(path, image_path)
#     image = Image.open(input_path)
#     image = image.resize((50, 50), Image.NEAREST)
#     image.save('smallFlowers/' + image_path)
#     i += 1
#     if i % 500 == 0:
#         print(i)

## Then save them all to a numpy array for use and easier access again
# images = []
#
# path = './smallFlowers'
# for image_path in os.listdir(path):
#     input_path = os.path.join(path, image_path)
#     image = misc.imread(input_path)
#     images.append(image)
# x_train = np.array(images)
# np.save("images.npy", x_train)

#x_train = np.load('images.npy')
#print(x_train.shape)

# import the data
(x_train, _), (_, _) = cifar10.load_data()

#reduce the inputs to be in range [0, 1]
x_train = x_train / 255

class I_Think_I_GAN:
    def __init__(self, dim=100):
        self.input_dim = dim    #dimension of input noise vector

    def inputs(self):
        # define the input placeholders
        gen_input = tf.placeholder(shape=(None, self.input_dim), dtype=tf.float32)
        real_input = tf.placeholder(shape=(None, 32, 32, 3), dtype=tf.float32)

        return gen_input, real_input

    def get_batch(self, data, batch_size=128):
        iterations = data.shape[0] // batch_size
        # mix up the images
        np.random.shuffle(x_train)
        # split them into batches
        for batch in np.array_split(x_train[:iterations*batch_size], iterations):
            yield batch * 2 - 1  # scale to -1 to 1

    def get_optimizer(self, lr=0.0005, beta1=0.5, beta2=0.999):
        # reset the default graph
        tf.reset_default_graph()
        # setup the inputs
        fake_input, real_input = self.inputs()
        # generate the generator
        fake_images = self.generator(input_layer=fake_input, reuse=False, lrelu_slope=0.2, training=True)
        # create the discriminator for the real images
        real_discrim = self.discriminator(input_layer=real_input, reuse=False, lrelu_slope=0.2)
        # create the discriminator for fake images
        fake_discrim = self.discriminator(input_layer=fake_images, reuse=True, lrelu_slope=0.2)
        # generator loss
        gen_loss = tf.reduce_mean(tf.losses.sigmoid_cross_entropy(logits=fake_discrim, multi_class_labels=tf.ones_like(fake_discrim)))
        # discriminator loss
        real_loss = tf.reduce_mean(tf.losses.sigmoid_cross_entropy(logits=real_discrim, multi_class_labels=tf.ones_like(real_discrim)))
        fake_loss = tf.reduce_mean(tf.losses.sigmoid_cross_entropy(logits=fake_discrim, multi_class_labels=tf.zeros_like(fake_discrim)))
        discrim_loss = real_loss + fake_loss
        # get the variables for the generator and discriminator
        gen_vars = [var for var in tf.trainable_variables() if var.name.startswith('generator')]
        discrim_vars = [var for var in tf.trainable_variables() if var.name.startswith('discriminator')]
        # setup the optimizers
        with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
            gen_optimizer = tf.train.AdamOptimizer(learning_rate=lr, beta1=beta1, beta2=beta2).minimize(gen_loss, var_list=gen_vars)
            discrim_optimizer = tf.train.AdamOptimizer(learning_rate=lr, beta1=beta1, beta2=beta2).minimize(discrim_loss, var_list=discrim_vars)

        return discrim_optimizer, gen_optimizer, fake_input, real_input

    def generator(self, input_layer, reuse=False, lrelu_slope=0.2, training=True):
        with tf.variable_scope('generator', reuse=reuse):
            # first layer
            input_dense = tf.layers.dense(inputs=input_layer, units=2*2*256)
            input_volume = tf.reshape(tensor=input_dense, shape=(-1, 2, 2, 256))
            layer1 = tf.layers.batch_normalization(inputs=input_volume, training=training)
            layer1 = tf.maximum(layer1 * lrelu_slope, layer1)
            # second layer
            layer2 = tf.layers.conv2d_transpose(filters=128, strides=2, kernel_size=5, padding='same', inputs=layer1)
            layer2 = tf.layers.batch_normalization(inputs=layer2, training=training)
            layer2 = tf.maximum(layer2 * lrelu_slope, layer2)
            # third layer
            layer3 = tf.layers.conv2d_transpose(filters=64, strides=2, kernel_size=5, padding='same', inputs=layer2)
            layer3 = tf.layers.batch_normalization(inputs=layer3, training=training)
            layer3 = tf.maximum(layer3 * lrelu_slope, layer3)
            # fourth layer
            layer4 = tf.layers.conv2d_transpose(filters=32, strides=2, kernel_size=5, padding='same', inputs=layer3)
            layer4 = tf.layers.batch_normalization(inputs=layer4, training=training)
            layer4 = tf.maximum(layer4 * lrelu_slope, layer4)
            # final layer
            logits = tf.layers.conv2d_transpose(filters=3, strides=2, kernel_size=5, padding='same', inputs=layer4)
            # output image
            out = tf.tanh(x=logits)

            return out

    def discriminator(self, input_layer, reuse=False, lrelu_slope=0.2):
        with tf.variable_scope('discriminator', reuse=reuse):
            # first layer
            layer1 = tf.layers.conv2d(inputs=input_layer, filters=32, strides=2, kernel_size=5, padding='same')
            layer1 = tf.maximum(layer1 * lrelu_slope, layer1)
            # second layer
            layer2 = tf.layers.conv2d(inputs=layer1, filters=64, strides=2, kernel_size=5, padding='same')
            layer2 = tf.layers.batch_normalization(inputs=layer2, training=True)
            layer2 = tf.maximum(layer2 * lrelu_slope, layer2)
            # third layer
            layer3 = tf.layers.conv2d(inputs=layer2, filters=128, strides=2, kernel_size=5, padding='same')
            layer3 = tf.layers.batch_normalization(inputs=layer3, training=True)
            layer3 = tf.maximum(layer3 * lrelu_slope, layer3)
            #fourth layer
            layer4 = tf.layers.conv2d(inputs=layer3, filters=256, strides=2, kernel_size=5, padding='same')
            layer4 = tf.layers.batch_normalization(inputs=layer4, training=True)
            layer4 = tf.maximum(layer4 * lrelu_slope, layer4)
            # flatten the array
            flatten = tf.reshape(tensor=layer4, shape=(-1, 2*2*256))
            # final layer
            final = tf.layers.dense(inputs=flatten, units=1)
            return final

    def train(self, batch_size=128, epochs=100):
        disc_optimizer, gen_optimizer, fake_input, real_input = self.get_optimizer()

        # run the training session
        with tf.Session() as sess:
            # initialize the variables
            sess.run(tf.global_variables_initializer())
            # train the network
            for epoch in tqdm(range(epochs)):
                for step, batch in enumerate(self.get_batch(x_train, batch_size)):
                    # get random noise vector
                    noise = np.random.uniform(low=-1, high=1, size=(batch_size, self.input_dim))
                    # run the generator
                    sess.run(gen_optimizer, feed_dict={fake_input: noise, real_input: batch})
                    # run the discriminator
                    sess.run(disc_optimizer, feed_dict={fake_input: noise, real_input: batch})

                # save a generated image every 5 epochs
                if epoch % 5 == 0:
                    # get some random noise vector
                    noise = np.random.uniform(low=-1, high=1, size=(100, self.input_dim))
                    # generate images
                    images = sess.run(self.generator(fake_input, reuse=True, training=False), feed_dict={fake_input: noise})
                    # save the images
                    self.plot_generated_images(epoch, images)

    def plot_generated_images(self, epoch, images, examples=100, dim=(10, 10), figsize=(10, 10)):
        plt.figure(figsize=figsize)
        # fit the images in the range 0 to 1 instead of -1, 1
        images = (images * .5) + .5
        for i in range(images.shape[0]):
            plt.subplot(dim[0], dim[1], i+1)
            plt.imshow(images[i], interpolation='nearest', cmap='gray')
            plt.axis('off')
        plt.tight_layout()
        plt.savefig('gan_generated_image_epoch_%d.png' % epoch)

gan = I_Think_I_GAN()
gan.train(batch_size=128, epochs=100)
