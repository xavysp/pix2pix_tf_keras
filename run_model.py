from __future__ import absolute_import, division, print_function

import time
import numpy as np
from os.path import join
from matplotlib import pyplot as plt
import tensorflow as tf
tf.enable_eager_execution()

from model import *
from utls import data_parser, make_dirs


BUFFER_SIZE = 400
# tf.set_random_seed(1)
class run_gan():

    def __init__(self, args):
        self.model_state= args.model_state
        self.args = args
        self.img_width=args.image_width
        self.img_height = args.image_height
        self.epochs = args.max_epochs
        self.bs = args.batch_size

    def load(self,image_file):
        input_img = tf.io.read_file(image_file[0])

        input_img = tf.image.decode_jpeg(input_img)

        # opening target image in png
        real_img = tf.io.read_file(image_file[1])
        real_img=tf.image.decode_png(real_img)
        real_img = tf.cast(real_img, tf.float32)
        input_img = tf.cast(input_img, tf.float32)

        return input_img, real_img

    def resize(self,input_image,real_image):
        input_image = tf.image.resize(input_image, [self.img_height, self.img_width],
                                      method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
        real_image = tf.image.resize(real_image, [self.img_height, self.img_width],
                                     method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
        return input_image, real_image

    def random_crop(self, input_img,real_img):
        stacked_img = tf.stack([input_img, real_img],axis=0)
        cropped_img = tf.image.random_crop(stacked_img,size=[2,self.img_height,self.img_width,3])
        return cropped_img[0],cropped_img[1]

    def pix2pix_norm(self,input_img,real_img):
        input_img = (input_img/127.5)-1
        real_img = (real_img/127.5)-1
        return input_img, real_img

    # @tf.function()
    def random_jitter(self, input_img, real_img):
        input_img, real_img = self.random_crop(input_img,real_img)

        if np.random.random()>0.5:
            input_img = tf.image.flip_left_right(input_img)
            real_img = tf.image.flip_left_right(real_img)

        return input_img, real_img

    def load_train_img(self,input_paths, target_path):
        input_img, real_img = self.load([input_paths, target_path])
        input_img, real_img = self.random_jitter(input_img, real_img)
        input_img, real_img = self.pix2pix_norm(input_img, real_img)
        return input_img, real_img

    def load_test_img(self,image_paths):
        input_img, real_img = self.load(image_paths)
        input_img, real_img = self.pix2pix_norm(input_img, real_img)
        return input_img, real_img

    def generate_images(self, model,test_input, tar,fig=None):
        with self.val_writer.as_defaul():
            pred = model(test_input,training=True)

        display_list = [test_input[0],tar[0],pred[0]]
        title=['Input image', 'Grount truth', 'Predicted Image']

        for i in range(3):
            plt.subplot(1,3,i+1)
            plt.title(title[i])
            plt.imshow(display_list[i]*0.5+0.5)
            plt.axis('off')
        plt.draw()
        plt.pause(0.01)

    def visualize(self,input, tar, g_pred,d_pred):
        d_pred = d_pred[0]
        d_pred = tf.expand_dims(d_pred,axis=0)
        d_pred = tf.image.resize_bilinear(d_pred,size=[256,256])
        d_pred = tf.concat([d_pred, d_pred, d_pred], axis=3)
        d_pred = tf.squeeze(d_pred)
        display_list = [input[0], tar[0], g_pred[0],d_pred]
        title = ['Input image', 'Grount truth', 'G Prediction', 'D Prediction']

        for i in range(4):
            plt.subplot(1, 4, i + 1)
            plt.title(title[i])
            plt.imshow(display_list[i] * 0.5 + 0.5)
            plt.axis('off')
        plt.draw()
        plt.pause(0.01)


    def train(self):
        # Validation and Train dataset generation

        data_cache =  data_parser(dataset_dir=self.args.data_base_dir,dataset_name=self.args.data4train,
                                 list_name=self.args.train_list)
        # Training
        train_list = data_cache['train_paths']
        train_list = np.array(train_list)
        train_data = tf.data.Dataset.from_tensor_slices((train_list[:,0],train_list[:,1]))
        train_data = train_data.shuffle(BUFFER_SIZE)
        train_data = train_data.map(self.load_train_img, num_parallel_calls=tf.data.experimental.AUTOTUNE)
        train_data = train_data.batch(self.bs)
        train_data = train_data.prefetch(1)
        # Validation
        val_list = data_cache['val_paths']
        val_list = np.array(val_list)
        val_data = tf.data.Dataset.from_tensor_slices((val_list[:,0],val_list[:,1]))
        val_data = val_data.map(self.load_train_img)
        val_data = val_data.batch(self.bs)
        val_data = val_data.prefetch(1)
        # call models
        G = Generator()  # Generator initialization
        D = Discriminator()  # Discriminator initialization

        # set and define optimizers
        G_optimizer = tf.train.AdamOptimizer(learning_rate=self.args.lr, beta1=self.args.beta1)
        D_optimizer = tf.train.AdamOptimizer(learning_rate=2e-4, beta1=0.5)
        # check previous training
        checkpoint_dir = join(self.args.checkpoint_dir,
                              join(self.args.model_name + '_' + self.args.data4train,
                                   self.args.model_state))
        _ = make_dirs(checkpoint_dir)
        checkpoint = tf.train.Checkpoint(G_optimizer=G_optimizer, D_optimizer=D_optimizer,
                                         G=G, D=D)
        # create summary
        train_log_dir = join('logs', self.args.model_name.lower() +
                                  '_' + self.args.dataset4training.lower(), self.args.model_state)
        val_log_dir = join('logs', self.args.model_name.lower() +
                                '_' + self.args.dataset4training.lower(), 'val')
        self.train_writer = tf.contrib.summary.create_file_writer(train_log_dir,flush_millis=10)
        self.val_writer = tf.contrib.summary.create_file_writer(val_log_dir, flush_millis=10)
        # start training
        plt.figure(figsize=(10,5))
        for epoch in range(self.epochs):
            start_time = time.time()
            iter = 0
            for input_img, target in train_data:
                with tf.GradientTape() as G_tape, tf.GradientTape() as D_tape:

                    with self.train_writer.as_defaul():

                        g_output = G(inputs=input_img,training=True)
                        d_real_output = D(inputs=[input_img,target],training=True)
                        d_gen_output = D(inputs=[input_img,g_output],training=True)

                    g_loss= G_loss(disc_generated_output=d_gen_output,gen_output=g_output,
                                              target=target)
                    d_loss = D_loss(disc_real_output=d_real_output,
                                               disc_generated_output=d_gen_output)
                g_grads = G_tape.gradient(g_loss,G.variables)
                d_grads = D_tape.gradient(d_loss,D.variables)

                G_optimizer.apply_gradients((zip(g_grads,G.variables)))
                D_optimizer.apply_gradients((zip(d_grads,D.variables)))

                if iter%self.args.display_freq==0:
                    #validation and visualization
                    self.visualize(input_img,target,g_output,d_gen_output)

                print('Itearacion: {} g_loss {}, d_loss {}'.format(iter,g_loss,d_loss))
                iter+=1

            for inp, tar in val_data.take(1):
                self.generate_images(G, inp, tar)
            if epoch % self.args.save_freq == 0:
                # saving checkpoint in every 20 epoch
                checkpoint.save(file_prefix=checkpoint_dir)

            print('Time taken for epoch {} is {} sec '.format(epoch + 1,
                                                               time.time() - start_time),
                  'g_loss {}, d_loss {}'.format(g_loss,d_loss))

    def test(self):
        # test dataset generation
        data_cache =  data_parser(dataset_dir=self.args.data_base_dir,dataset_name=self.args.data4test,
                                 list_name=self.args.test_list,is_train=False)
        data_list = data_cache['files_path']
        test_data = tf.data.Dataset.list_files(data_list)
        test_data = test_data.shuffle(BUFFER_SIZE)
        test_data = test_data.map(self.load_test_img)
        test_data = test_data.batch(self.bs)
        # call models
        # self.model = define_model(args=self.args)
        G = Generator() # Generator initialization
        D = Discriminator() # Discriminator initialization

        #set and define optimizers
        G_optimizer = tf.train.AdamOptimizer(learning_rate=2e-4,beta1=0.5)
        D_optimizer = tf.train.AdamOptimizer(learning_rate=2e-4,beta1=0.5)
        # check previous training
        checkpoint_dir = join(self.args.checkpoint_dir,
                                      join(self.args.model_name + '_' + self.args.data4train,
                                                   self.args.model_state))
        checkpoint = tf.train.Checkpoint(G_optimizer=G_optimizer,D_optimizer=D_optimizer,
                                         G=G,D=D)