#!/usr/bin/env python
# coding: utf-8

# In[139]:


GPU = False
if GPU:
	import os
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
from keras.models import Model, load_model
from keras import metrics
from keras import backend as K   # 'generic' backend so code works with either tensorflow or theano
from keras.callbacks import ModelCheckpoint

from sys import exit
import tensorflow as tf
import scipy as sp

from sklearn.manifold import TSNE

import pickle


# In[84]:


Z_train = pd.read_csv("train.csv")
# print(Z_train)


# In[24]:


Z_train = Z_train


# In[27]:


# Z_train.describe()


# In[142]:


y_train = Z_train['label'].values[:10000]
X_train = Z_train.drop('label', axis=1).values[:10000]

y_train = y_train.reshape(-1, 1)
X_train = X_train.reshape(X_train.shape[0], 28, 28, 1) / 255


# In[143]:


X_val = Z_train.drop('label',axis=1).values[40000:]
X_val = X_val.reshape(X_val.shape[0], 28, 28, 1) / 255
y_val = Z_train['label'][40000:]


# In[152]:


# set up encoder

img_shape = (28, 28, 1)    # for MNIST
batch_size = 16
latent_dim = 10  # Number of latent dimension parameters

# sampling function
def sampling(args):
    latent_dim = 10  # Number of latent dimension parameters

    z_mu, z_log_sigma = args

    # sample from N(0,1)
    epsilon = K.random_normal(shape=(K.shape(z_mu)[0], latent_dim),
                              mean=0., stddev=1.)

    # convert to x ~ N(z_mu, z_sigma)
    return z_mu + K.exp(z_log_sigma) * epsilon

input_img = keras.Input(shape=img_shape)

def get_encoder():
    # Encoder architecture: Input -> Conv2D*4 -> Flatten -> Dense

    x = layers.Conv2D(32, 3,
                      padding='same', 
                      activation='relu')(input_img)
    x = layers.Conv2D(64, 3,
                      padding='same', 
                      activation='relu',
                      strides=(2, 2))(x)
    x = layers.Conv2D(64, 3,
                      padding='same', 
                      activation='relu')(x)
    x = layers.Conv2D(64, 3,
                      padding='same', 
                      activation='relu')(x)
    # need to know the shape of the network here for the decoder
    shape_before_flattening = K.int_shape(x)

    x = layers.Flatten()(x)
    x = layers.Dense(32, activation='relu')(x)

    # Two outputs, latent mean and (log)variance
    z_mu = layers.Dense(latent_dim)(x)
    z_log_sigma = layers.Dense(latent_dim)(x)
    

    # sample vector from the latent distribution
    z = layers.Lambda(sampling)([z_mu, z_log_sigma])
    
    return z, z_mu, z_log_sigma,shape_before_flattening


# In[ ]:


z, z_mu, z_log_sigma, shape_before_flattening = get_encoder()


# In[146]:


# set up decoder

# decoder takes the latent distribution sample as input
def get_decoder():
    decoder_input = layers.Input(K.int_shape(z)[1:])

    # Expand to 784 total pixels
    x = layers.Dense(np.prod(shape_before_flattening[1:]),
                     activation='relu')(decoder_input)

    # reshape
    x = layers.Reshape(shape_before_flattening[1:])(x)

    # use Conv2DTranspose to reverse the conv layers from the encoder
    x = layers.Conv2DTranspose(32, 3,
                               padding='same', 
                               activation='relu',
                               strides=(2, 2))(x)
    x = layers.Conv2D(1, 3,
                      padding='same', 
                      activation='sigmoid')(x)

    # decoder model statement
    decoder = Model(decoder_input, x)
    
    return decoder


# In[147]:


decoder = get_decoder()
# apply the decoder to the sample from the latent distribution
z_decoded = decoder(z)


# In[148]:


decoder.summary()


# apply the custom loss to the input images and the decoded latent distribution sample


# In[109]:

def vae_loss(x, z_decoded):
    x = K.flatten(x)
    z_decoded = K.flatten(z_decoded)
    # Reconstruction loss
    xent_loss = keras.metrics.binary_crossentropy(x, z_decoded)
    # KL divergence
    kl_loss = -5e-4 * K.mean(1 + z_log_sigma - K.square(z_mu) - K.exp(z_log_sigma), axis=-1)
    return tf.reshape(K.mean(xent_loss + kl_loss), (1,-1))





# In[150]:
print("loading model....")

# K.clear_session()

# vae = load_model("weights_mnist.01-0.26.ckpt", custom_objects={ 'vae_loss': vae_loss})
print("LOADED!")

# VAE model statement
vae = Model(input_img, z_decoded)
vae.compile(optimizer='adam', loss=vae_loss)
vae.load_weights('weights_mnist.01-0.25.ckpt')
vae.summary()


# In[151]:



filepath = "model.mnist.ckpt"
filepath = "weights_mnist.{epoch:02d}-{val_loss:.2f}.ckpt"
checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=1, 
                            save_best_only=False, save_weights_only=False,mode='auto', period=1)
callbacks_list = [checkpoint]


print(X_train.shape)
# exit(0)

vae.fit(x=X_train, y=X_train,
        shuffle=True,
        epochs=2,
        batch_size=batch_size,
        callbacks=callbacks_list,
        validation_data=(X_val, X_val))

# exit(0)

# In[ ]:


encoder = Model(input_img, z_mu)
z_mu_new = encoder.predict(X_val, batch_size=batch_size)

pickle.dump(z_mu_new, "z_mu.p")
pickle.dump(y_val, "y_val.p")



# In[114]:


"""
# Display a 2D manifold of the digits
n = 20  # figure with 20x20 digits
digit_size = 28
figure = np.zeros((digit_size * n, digit_size * n))

# Construct grid of latent variable values
grid_x = norm.ppf(np.linspace(0.05, 0.95, n))
grid_y = norm.ppf(np.linspace(0.05, 0.95, n))

# decode for each square in the grid
for i, yi in enumerate(grid_x):
    for j, xi in enumerate(grid_y):
        z_sample = np.array([[xi, yi]])
        z_sample = np.tile(z_sample, batch_size).reshape(batch_size, 2)
        x_decoded = decoder.predict(z_sample, batch_size=batch_size)
        digit = x_decoded[0].reshape(digit_size, digit_size)
        figure[i * digit_size: (i + 1) * digit_size,
               j * digit_size: (j + 1) * digit_size] = digit

plt.figure(figsize=(10, 10))
plt.imshow(figure, cmap='gnuplot2')
plt.show() 
"""

# In[ ]:




