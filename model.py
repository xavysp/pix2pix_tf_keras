
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
tk = tf.keras

def down_sample(n_filters, k_size,use_bn=True):
    initializer=tf.random_normal_initializer(mean=0.,stddev=0.02)
    result = tk.Sequential()
    result.add(tk.layers.Conv2D(filters=n_filters,kernel_size=k_size,strides=2,padding='same',
                                kernel_initializer=initializer,use_bias=False))
    if use_bn:
        result.add(tk.layers.BatchNormalization())
    result.add(tk.layers.LeakyReLU())
    return result

def up_sample(n_filters, k_size,use_dropout=False, use_bn=True):
    initializer = tf.random_normal_initializer(mean=0., stddev=0.02)
    result = tk.Sequential()
    result.add(tk.layers.Conv2DTranspose(filters=n_filters,kernel_size=k_size,strides=2,
                                         padding='same',kernel_initializer=initializer, use_bias=False))
    if use_bn:
        result.add(tk.layers.BatchNormalization())
    if use_dropout:
        result.add(tk.layers.Dropout(0.5))
    result.add(tk.layers.ReLU())
    return result

# @tf.contrib.eager.defun
def Generator():
    # based un U-net
    down_stack = [
        down_sample(n_filters=64,k_size=4,use_bn=False), #[BS,128,128,64]
        down_sample(n_filters=128,k_size=4), #[BS,64,64,128]
        down_sample(n_filters=256,k_size=4), #[BS,32,32,256]
        down_sample(n_filters=512,k_size=4), #[BS,16,16,512]
        down_sample(n_filters=512,k_size=4), #[BS,8,8,512]
        down_sample(n_filters=512,k_size=4), #[BS,4,4,512]
        down_sample(n_filters=512,k_size=4), #[BS,2,2,512]
        down_sample(n_filters=512,k_size=4), #[BS,1,1,512]
    ]

    up_stack =[
        up_sample(n_filters=512,k_size=4,use_dropout=True,use_bn=True), # [BS,2,2,1024]
        up_sample(n_filters=512,k_size=4,use_dropout=True,use_bn=True), # [BS,4,4,1024]
        up_sample(n_filters=512,k_size=4,use_dropout=True,use_bn=True), # [BS,8,8,1024]
        up_sample(n_filters=512,k_size=4), # [BS,16,16,1024]
        up_sample(n_filters=256,k_size=4), # [BS,32,32,512]
        up_sample(n_filters=128,k_size=4), # [BS,64,64,256]
        up_sample(n_filters=64,k_size=4), # [BS,128,128,128]
    ]

    initializer = tf.random_normal_initializer(mean=0.,stddev=0.02)
    last_up = tk.layers.Conv2DTranspose(filters=3,kernel_size=4, strides=2,
                                        padding='same', kernel_initializer=initializer,
                                        activation='tanh') # [BS, 256,256,3]
    concat = tk.layers.Concatenate()
    inputs = tk.layers.Input(shape=[None,None,3])
    x= inputs
     # check this in the debuging
    skips = []
    for down in down_stack:
        x = down(x)
        skips.append(x)
    skips = reversed(skips[:-1])

    # up samppling and establishing the skip connections
    for up, skip in zip(up_stack, skips):
        x = up(x)
        x = concat([x, skip])
    # below the backward process
    x = last_up(x)

    return tk.Model(inputs=inputs, outputs=x)

# @tf.contrib.eager.defun
def Discriminator():
    initializer = tf.random_normal_initializer(mean=0.0,stddev=0.02)

    inp = tk.layers.Input(shape=[None,None,3], name = 'input_image')
    tar = tk.layers.Input(shape = [None,None,3], name = 'target_image')

    x = tk.layers.concatenate([inp,tar]) # [BS,256,256,[inp_channels+tar_channels]]

    down1 = down_sample(n_filters=64,k_size=4,use_bn=False)(x) # [BS,128,128,64]
    down2 = down_sample(n_filters=128,k_size=4,use_bn=True)(down1) # [BS,64,64,128]
    down3 = down_sample(n_filters=256,k_size=4,use_bn=True)(down2) # [BS,32,32,256]

    zero_pad1 = tk.layers.ZeroPadding2D()(down3) # [BS,34,34,256]
    conv = tk.layers.Conv2D(512,4,strides=1,kernel_initializer = initializer,
                            use_bias=False)(zero_pad1) # [BS,31,31,512]
    bn1 = tk.layers.BatchNormalization()(conv)
    leaky_relu = tk.layers.LeakyReLU()(bn1)

    zero_pad2 = tk.layers.ZeroPadding2D()(leaky_relu) # [BS,33,33,512]
    last = tk.layers.Conv2D(1, 4, strides=1,kernel_initializer=initializer)(zero_pad2) # [BS,30,30,1]

    return  tk.Model(inputs=[inp,tar],outputs=last)

def D_loss(disc_real_output,disc_generated_output):

    loss_object = tk.losses.BinaryCrossentropy(from_logits=True)
    real_loss = loss_object(tf.ones_like(disc_real_output),disc_real_output)

    generated_loss = loss_object(tf.zeros_like(disc_generated_output), disc_generated_output)

    total_disc_loss = real_loss + generated_loss
    tf.contrib.summary.scalar('G_loss',total_disc_loss)
    return total_disc_loss

def G_loss(disc_generated_output, gen_output,target):
    lmbda = 100
    loss_object = tk.losses.BinaryCrossentropy(from_logits=True)
    gan_loss = loss_object(tf.ones_like(disc_generated_output), disc_generated_output)

    # mean absolute error
    l1_loss = tf.reduce_mean(tf.abs(target-gen_output))

    total_gen_loss = gan_loss+(lmbda*l1_loss)
    tf.contrib.summary.scalar('D_loss',total_gen_loss)
    return total_gen_loss