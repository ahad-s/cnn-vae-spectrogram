#!/usr/bin/env python
# coding: utf-8

GPU = False
import os

if GPU:

  os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
  os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from scipy.stats import norm

import keras
from keras import layers
from keras.layers import MaxPooling2D, Conv2D, Dense, Softmax, Flatten
from keras.models import Model
from keras import metrics
from keras import backend as K   # 'generic' backend so code works with either tensorflow or theano

from sys import exit
import tensorflow as tf
import scipy as sp

from sklearn.manifold import TSNE

import random
from PIL import Image
from keras.callbacks import ModelCheckpoint
import pickle



img_shape = (500, 500, 4)
batch_size = 16
latent_dim = 50  # Number of latent dimension parameters
epochs = 10
# In[8]:


folder = "spectrogram_images/"
im_names = os.listdir(folder)
random.shuffle(im_names)
# im_names_train = random.choices(im_names, k=len(im_names))
X = []
y = []

# convert artist to number here
artist_num = 0
artists = {}

for name in im_names:
    pic = Image.open(folder + name)
    artist_name = name.split("_")[0]
    
    if artist_name in artists:
        y.append(artists[artist_name])
    else:
        artists[artist_name] = artist_num
        y.append(artist_num)
        
        artist_num += 1
        
    x_i = np.array(pic)
    X.append(x_i)
    
X = np.array(X)
y = np.array(y)

val_pct = 0.1
train_num = (1-val_pct)*len(X)
X_train = X[:train_num] / 255
y_train = y[:train_num]

X_val = X[train_num:] / 255
y_val = y[train_num:]


# In[10]:


# ENCODER

# Encoder architecture: VGG16 encoder with VGG16-style DCNN decoder
input_img = keras.Input(shape=img_shape)

x = Conv2D(64, (3,3), strides=1, padding='same',
    input_shape=(500, 500, 4), 
    activation='relu')(input_img)

x = Conv2D(64, (3,3), strides=1, padding='same',
    activation='relu')(x)

###############################################
x = MaxPooling2D(pool_size=(2,2))(x)
###############################################

x = Conv2D(128, (3,3), strides=1, padding='same', 
  activation='relu',)(x)

x = Conv2D(128, (3,3), strides=1, padding='same', 
  activation='relu',)(x)

###############################################
x = MaxPooling2D(pool_size=(2,2))(x)
###############################################

x = Conv2D(256, (3,3), strides=1, padding='same',
  activation='relu')(x)

x = Conv2D(256, (3,3), strides=1, padding='same',
  activation='relu')(x)

x = Conv2D(256, (3,3), strides=1, padding='same',
  activation='relu')(x)

###############################################
x = MaxPooling2D(pool_size=(2,2))(x)
###############################################

x = Conv2D(512, (3,3), strides=1, padding='same',
  activation='relu')(x)

x = Conv2D(512, (3,3), strides=1, padding='same',
  activation='relu')(x)

x = Conv2D(512, (3,3), strides=1, padding='same',
  activation='relu')(x)

###############################################
x = MaxPooling2D(pool_size=(2,2))(x)
###############################################

x = Conv2D(512, (3,3), strides=1, padding='same',
  activation='relu')(x)
x = Conv2D(512, (3,3), strides=1, padding='same',
  activation='relu')(x)
x = Conv2D(512, (3,3), strides=1, padding='same',
  activation='relu')(x)

###############################################
x = MaxPooling2D(pool_size=(2,2))(x)
###############################################

shape_before_flattening = K.int_shape(x)

x = Flatten()(x)

x = Dense(4096, activation='relu')(x)
x = Dense(4096, activation='relu')(x)

# Two outputs, latent mean and (log)variance
z_mu = Dense(latent_dim)(x)
z_log_sigma = Dense(latent_dim)(x)

# sampling function
def sampling(args):
    z_mu, z_log_sigma = args

    # sample from N(0,1)
    epsilon = K.random_normal(shape=(K.shape(z_mu)[0], latent_dim),
                              mean=0., stddev=1.)

    # convert to x ~ N(z_mu, z_sigma)
    return z_mu + K.exp(z_log_sigma) * epsilon

# sample vector from the latent distribution

z = layers.Lambda(sampling)([z_mu, z_log_sigma])


# In[11]:


# DECODER

# decoder takes the latent distribution sample as input
decoder_input = layers.Input(K.int_shape(z)[1:])

# Expand to 784 total pixels
x = layers.Dense(np.prod(shape_before_flattening[1:]),
                 activation='relu')(decoder_input)

# reshape
x = layers.Reshape(shape_before_flattening[1:])(x)

# use Conv2DTranspose to reverse the conv layers from the encoder

x = Conv2D(512, (3,3), strides=1, padding='same',
  activation='relu')(x)

x = Conv2D(512, (3,3), strides=1, padding='same',
  activation='relu')(x)

###############################################
x = layers.Conv2DTranspose(512, 3,
                           padding='valid', 
                           activation='relu',
                           strides=(2, 2))(x)
###############################################


x = Conv2D(512, (3,3), strides=1, padding='same',
  activation='relu')(x)

x = Conv2D(512, (3,3), strides=1, padding='same',
  activation='relu')(x)


###############################################
x = layers.Conv2DTranspose(512, 3,
                           padding='same', 
                           activation='relu',
                           strides=(2, 2))(x)
###############################################


x = Conv2D(256, (3,3), strides=1, padding='same',
  activation='relu')(x)

x = Conv2D(256, (3,3), strides=1, padding='same',
  activation='relu')(x)

###############################################
x = layers.Conv2DTranspose(256, 3,
                           padding='valid', 
                           activation='relu',
                           strides=(2, 2))(x)
###############################################


x = Conv2D(128, (3,3), strides=1, padding='same', 
  activation='relu',)(x)

x = layers.Conv2DTranspose(128, 3,
                           padding='same', 
                           activation='relu',
                           strides=(2, 2))(x)

x = Conv2D(64, (3,3), strides=1, padding='same', 
  activation='relu',)(x)

###############################################
x = layers.Conv2DTranspose(64, 3,
                           padding='same', 
                           activation='relu',
                           strides=(2, 2))(x)
###############################################

x = layers.Conv2D(4, 3,
                  padding='same', 
                  activation='sigmoid')(x)

# decoder model statement
decoder = Model(decoder_input, x)

# apply the decoder to the sample from the latent distribution
z_decoded = decoder(z)


# In[12]:


decoder.summary()


# In[13]:


def vae_loss(x, z_decoded):
    x = K.flatten(x)
    z_decoded = K.flatten(z_decoded)
    # Reconstruction loss
    xent_loss = keras.metrics.binary_crossentropy(x, z_decoded)
    # KL divergence
    kl_loss = -5e-4 * K.mean(1 + z_log_sigma - K.square(z_mu) - K.exp(z_log_sigma), axis=-1)
    return K.mean(xent_loss + kl_loss)


# In[16]:


# VAE model statement
vae = Model(input_img, z_decoded)
vae.compile(optimizer='adam', loss=vae_loss)
vae.summary()


# In[ ]:


filepath = "weights_vgg.{epoch:03d}-{val_loss:.2f}.ckpt"
checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=1, 
                            save_best_only=False, save_weights_only=True,mode='auto', period=1)
callbacks_list = [checkpoint]

vae.fit(x=X_train, y=X_train,
        shuffle=True,
        epochs=epochs,
        batch_size=batch_size,
#        validation_split=0.1)
        validation_data=(X_val, X_val))


# In[ ]:


encoder = Model(input_img, z_mu)
z_mu_new = encoder.predict(X_val, batch_size=batch_size)


dd = lambda d, f: pickle.dump(d, open(f, "wb"))
dd(d_mu_new, "z_mu.p")
dd(y_val, "y_val.p")
dd(artists, "artists.p")



# In[ ]:


"""

USE_TSNE=False

if USE_TSNE:
    reducer = TSNE(n_components=2, verbose=1)
    z_mu_2 = reducer.fit_transform(z_mu_new)
    plt.figure(figsize=(10, 10))
    plt.scatter(z_mu_2[:, 0], z_mu_2[:, 1], c=Z_train['label'], cmap='brg')
    plt.colorbar()
#     plt.show()
else:
    plt.figure(figsize=(10, 10))
    plt.scatter(z_mu_new[:, 0], z_mu_new[:, 1], c=y_train, cmap='brg')
    plt.colorbar()
#     plt.show()
    
plt.savefig("test.png")
"""

