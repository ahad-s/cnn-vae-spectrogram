#!/usr/bin/env python
# coding: utf-8

GPU = True
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
from keras.optimizers import Adam

from sys import exit
import tensorflow as tf
import scipy as sp

from sklearn.manifold import TSNE

import random
from PIL import Image
from keras.callbacks import ModelCheckpoint
import pickle


class CNN_VAE(object):
    def __init__(self):
      self.img_shape = (224, 224, 3)
      self.batch_size = 32
      self.latent_dim = 50  # Number of latent dimension parameters
      self.epochs = 1
      self.kl_lambda = 1
      self.artists = {}
      self.artist_num = 0
      self.val_pct = 0.5 # percent of validation
      self.learning_rate = 0.0001

      self.weight_file = None

      self.sp_folder = "spectrogram_images/"

      self.populate_images()


    def populate_images(self, n=100):
        im_names = os.listdir(self.sp_folder)
        if n != -1:
            im_names = im_names[:n]
        random.shuffle(im_names)
        # im_names_train = random.choices(im_names, k=len(im_names))
        X = []
        y = []
        for name in im_names:
            pic = Image.open(self.sp_folder + name)
            #pic = pic.convert('L')
            pic = pic.resize((224, 224))
            artist_name = name.split("_")[0]
            if artist_name in ['Nirvana', 'Rammstein', 'Soundgarden']:
                continue
            
            if artist_name in self.artists:
                y.append(self.artists[artist_name])
            else:
                self.artists[artist_name] = self.artist_num
                y.append(self.artist_num)
                
                self.artist_num += 1
                
            x_i = np.array(pic)
            X.append(x_i)
            #X.append(x_i.reshape(x_i.shape[0], x_i.shape[1], 1))
            
        print("Loaded %d images." % len(X))
        X = np.array(X)
        y = np.array(y)

        self.X = X[:, :, :, :-1]

        self.train_num = int((1-self.val_pct)*len(self.X))

        self.X_val = X[self.train_num:] / 255
        self.y_val = y[self.train_num:]

    def encoder(self, input_img):
        # Encoder architecture: VGG16 encoder with VGG16-style DCNN decoder

        x = Conv2D(64, (3,3), strides=1, padding='same',
            input_shape=(224, 224, 3), 
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

        self.shape_before_flattening = K.int_shape(x)

        x = Flatten()(x)

        x = Dense(4096, activation='relu')(x)
        x = Dense(4096, activation='relu')(x)

        # Two outputs, latent mean and (log)variance
        self.z_mu = Dense(self.latent_dim)(x)
        self.z_log_sigma = Dense(self.latent_dim)(x)

        # sampling function
        def sampling(args):
            z_mu, z_log_sigma = args

            # sample from N(0,1)
            epsilon = K.random_normal(shape=(K.shape(z_mu)[0], self.latent_dim),
                                      mean=0., stddev=1.)

            # convert to x ~ N(z_mu, z_sigma)
            return z_mu + K.exp(z_log_sigma) * epsilon

        # sample vector from the latent distribution

        self.z = layers.Lambda(sampling)([self.z_mu, self.z_log_sigma])

        return self.z, self.z_mu, self.z_log_sigma

    def decoder(self, decoder_input, z_enc):
        # decoder takes the latent distribution sample as input
        # decoder_input = layers.Input(K.int_shape(z)[1:])

        # Expand to 224x224 total pixels
        x = layers.Dense(np.prod(self.shape_before_flattening[1:]),
                         activation='relu')(decoder_input)

        # reshape
        x = layers.Reshape(self.shape_before_flattening[1:])(x)

        # use Conv2DTranspose to reverse the conv layers from the encoder

        x = Conv2D(512, (3,3), strides=1, padding='same',
          activation='relu')(x)

        #x = Conv2D(512, (3,3), strides=1, padding='same',
        #  activation='relu')(x)

        ###############################################
        x = layers.Conv2DTranspose(512, 3,
                                   padding='same', 
                                   activation='relu',
                                   strides=(2, 2))(x)
        ###############################################


        x = Conv2D(512, (3,3), strides=1, padding='same',
          activation='relu')(x)

        #x = Conv2D(512, (3,3), strides=1, padding='same',
        #  activation='relu')(x)


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
                                   padding='same', 
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

        x = layers.Conv2D(3, 3,
                          padding='same', 
                          activation='sigmoid')(x)

        # decoder model statement
        self.decoder = Model(decoder_input, x)

        # apply the decoder to the sample from the latent distribution
        self.z_decoded = self.decoder(z_enc)
        return self.z_decoded, self.decoder

    def build_model(self):
      self.input_img = keras.Input(shape=self.img_shape)
      z_enc, z_mu, z_log_sigma = self.encoder(self.input_img)
      self.decoder_input = layers.Input(K.int_shape(z_enc)[1:])
      z_dec, dec = self.decoder(self.decoder_input, z_enc)
      dec.summary()

      # VAE model statement
      opt = Adam(lr=self.learning_rate)
      self.vae = Model(self.input_img, z_dec)
      self.vae.compile(optimizer=opt, loss= self.vae_loss)
      if self.weight_file:
          vae.load_weights(self.weight_file)
      self.vae.summary()

      return self.vae


    def vae_loss(self, x, z_decoded):
        x = K.flatten(x)
        z_decoded = K.flatten(z_decoded)

        # Reconstruction loss
        xent_loss = keras.metrics.binary_crossentropy(x, z_decoded)
        xent_loss *= 224 * 224 * 3

        # KL divergence
        kl_loss = - self.kl_lambda * K.mean(1 + self.z_log_sigma - K.square(self.z_mu) - K.exp(self.z_log_sigma), axis=-1)
        return K.mean(xent_loss + kl_loss)

    def get_ckpointer(self):
        filepath = "weights_vgg.{epoch:03d}-{val_loss:.2f}.ckpt"
        checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=1, 
                                    save_best_only=False, save_weights_only=True,mode='auto', period=1)
        callbacks_list = [checkpoint]
        return callbacks_list

    def save_latent(self):
      encoder = Model(self.input_img, self.z_mu)
      z_mu_new = encoder.predict(self.X_val, batch_size=self.batch_size)


      dd = lambda d, f: pickle.dump(d, open(f, "wb"))
      dd(z_mu_new, "z_mu.p")
      dd(y_val, "y_val.p")
      dd(artists, "artists.p")


    def train(self):
        vae = self.build_model()

        print("STARTING EPOCHS...")

        def data_gen(X):
          n = len(X)
          i = 0

          while i < n:

            yield X[i:i+self.batch_size]/255
            i += batch_size

        old_vloss = 0
        err = 0.0001

        for e in range(self.epochs):
        #if False:
          i = 0
          for X_train in data_gen(self.X[:self.train_num]):
              i += self.batch_size
              if i > self.batch_size*5 and i % self.batch_size*5 == 0:
                  print(i)
              if (i > self.batch_size*30) and (i % (self.batch_size*30) == 0): # every 1k images, save latest
                hist = vae.fit(x=X_train, y=X_train,
                      shuffle=True,
                      epochs=1,
                      callbacks=callbacks_list,
                      batch_size=self.batch_size,
                      validation_data=(X_val, X_val))
               
                vloss = hist.history['val_loss'][-1]

                if abs(vloss - old_vloss) < err:
                    break
                old_vloss = vloss
              else:
                  vae.fit(x=X_train, y=X_train,
                      shuffle=True,
                      epochs=1,
                verbose=int(i % self.batch_size*5 == 0),
                      batch_size=self.batch_size)



model = CNN_VAE()
model.train()
model.save_latent()

