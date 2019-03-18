#!/usr/bin/env python
# coding: utf-8

r_seed = 12345
GPU = True
import os

if GPU:

  os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
  os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import pandas as pd
import numpy as np
np.random.seed(r_seed)
import tensorflow as tf
tf.set_random_seed(r_seed)
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from scipy.stats import norm, truncnorm

import keras
from keras import layers
from keras.layers import MaxPooling2D, Conv2D, Dense, Softmax, Flatten
from keras.models import Model
from keras import metrics
from keras import backend as K   # 'generic' backend so code works with either tensorflow or theano
from keras.optimizers import Adam

from sys import exit
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
      self.artists = {}
      self.artist_num = 0 # num_classes
      self.val_pct = 0.1 # percent of validation
      self.learning_rate = 0.000001

      self.kl_lambda = 1
      self.wae_lambda = 10
      self.recon_lambda = 1

      self.z_temp = 1 # temperature - multiplier for sigma

      self.prior_mu = 0
      self.prior_sigma = 1

      self.debug = False

      self.kernel = 'RBF' # Gaussian

      self.weight_file = None
#       self.weight_file = 'checkpoints/weights_vgg_wae_allartists.001-0.50.ckpt'
      #self.weight_file = 'checkpoints/weights_vgg_wae_allartists.001-0.51.ckpt'
      # self.weight_file = 'checkpoints/weights_vgg_wae_allartists_actual.001-0.52.ckpt'

      self.model = None 

      self.sp_folder = "spectrogram_images/"

      self.populate_images()
        
      self.z_prior = self.get_z_prior_samples()

#       self.get_z_prior_samples()

#       self.labels = 

      self.prior_means = []
      self.prior_sigmas = []
      self.prior_dists = []

    def get_z_prior(self):

      epsilon = K.random_normal(shape=(self.batch_size, self.latent_dim),
                                mean=0., stddev=1.)
      self.z_prior = self.prior_mu + self.prior_sigma * epsilon
      return self.z_prior


   def get_z_prior_samples(self):
      a, b, = -2, 2 # 2 stddevs away from N(0,1), as in tf.trunc_norm_initializer

      for i in range(self.artist_num):
        epsilon = K.random_normal([self.batch_size, self.latent_dim])
        mean = truncnorm(a, b).rvs(self.latent_dim)
        sigma = truncnorm(a, b).rvs(self.latent_dim)

        self.prior_dists_np.append((mean, sigma))

        mean = tf.convert_to_tensor(mean, name="prior_mean_{}".format(str(i)))
        sigma = tf.convert_to_tensor(sigma, name="prior_sigma_{}".format(str(i)))

        dist = K.cast(mean + epsilon * K.exp(sigma), dtype='int32')

        self.prior_means.append(mean)
        self.prior_sigmas.append(sigma)
        self.prior_dists.append(dist)
    
      print(self.prior_dists)
      print(self.input_labels)
      self.z_prior = tf.nn.embedding_lookup(self.prior_dists, self.input_labels)
      return self.z_prior



    def populate_images(self, n=-1):
        im_names = os.listdir(self.sp_folder)
        if n != -1:
            im_names = im_names[:n]
        random.shuffle(im_names)
        # im_names_train = random.choices(im_names, k=len(im_names))
        X = []
        y = []
        for name in im_names:
            artist_name = name.split("_")[0]
            if artist_name in ['Nirvana', 'Rammstein', 'Soundgarden']:
                continue

            #if artist_name not in ['DM', 'NeilYoung']:
            #   continue
            #if artist_name not in ['Doors', 'NIN']:
            #   continue

            pic = Image.open(self.sp_folder + name)
            pic = pic.resize((224, 224))

            if artist_name in self.artists:
                y.append(self.artists[artist_name])
            else:
                self.artists[artist_name] = self.artist_num
                y.append(self.artist_num)
                
                self.artist_num += 1
                
            x_i = np.array(pic)
            X.append(x_i)


        print("Loaded %d images." % len(X))

        X = np.array(X)
        y = np.array(y, dtype=np.int32)

        self.X = X[:, :, :, :-1]
        self.y = y

        self.train_num = int((1-self.val_pct)*len(self.X))

        self.X_val = self.X[self.train_num + 25:] / 255
        self.y_val = y[self.train_num + 25:]

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
#         self.z = self.sample_z_tilda_from_posterior()

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
        self.decoder_model = Model(decoder_input, x)

        # apply the decoder to the sample from the latent distribution
        self.z_decoded = self.decoder_model(z_enc)
        return self.z_decoded, self.decoder_model

    def build_model(self, loss=None):
      if loss is None:
        loss = self.vae_loss

      if loss is self.vae_loss:
          print("Using VAE LOSS")
      else:
          print("Using WAE LOSS")

      self.input_labels = keras.Input(shape=(1,), dtype='int32')
      self.input_img = keras.Input(shape=self.img_shape)

      self.get_z_prior_samples()

      z_enc, z_mu, z_log_sigma = self.encoder(self.input_img)
      self.decoder_input = layers.Input(K.int_shape(z_enc)[1:])
      z_dec, dec = self.decoder(self.decoder_input, z_enc)
      dec.summary()

      # VAE model statement
      opt = Adam(lr=self.learning_rate)
      vae = Model([self.input_img, self.input_labels], z_dec)
      vae.compile(optimizer=opt, loss=loss)
      if self.weight_file is not None:
          vae.load_weights(self.weight_file)
      vae.summary()
      self.model = vae
      return self.model


    # sample_pz -- p(z)
    # sample_qz -- q(z|x_i)
    def mmd_penalty(self, sample_qz, sample_pz):
        n = self.batch_size
        n = tf.cast(n, tf.int32)
        nf = tf.cast(n, tf.float32)
        half_size = tf.cast((n * n - n) / 2, tf.int32)

        norms_pz = tf.reduce_sum(tf.square(sample_pz), axis=1)
        dotprods_pz = tf.matmul(sample_pz, sample_pz, transpose_b=True)
        distances_pz = norms_pz + tf.transpose(norms_pz) - 2. * dotprods_pz

        norms_qz = tf.reduce_sum(tf.square(sample_qz), axis=1)
        dotprods_qz = tf.matmul(sample_qz, sample_qz, transpose_b=True)
        distances_qz = norms_qz + tf.transpose(norms_qz) - 2. * dotprods_qz

        dotprods = tf.matmul(sample_qz, sample_pz, transpose_b=True)
        distances = norms_qz + tf.transpose(norms_pz) - 2. * dotprods

        if self.kernel == 'RBF':
            # Median heuristic for the sigma^2 of Gaussian kernel
            sigma2_k = tf.nn.top_k(
                tf.reshape(distances, [-1]), half_size).values[half_size - 1]
            sigma2_k += tf.nn.top_k(
                tf.reshape(distances_qz, [-1]), half_size).values[half_size - 1]
            # if opts['verbose']:
            #     sigma2_k = tf.Print(sigma2_k, [sigma2_k], 'Kernel width:')
            res1 = tf.exp(- distances_qz / 2. / sigma2_k)
            res1 += tf.exp(- distances_pz / 2. / sigma2_k)
            res1 = tf.multiply(res1, 1. - tf.eye(n))
            res1 = tf.reduce_sum(res1) / (nf * nf - nf)
            res2 = tf.exp(- distances / 2. / sigma2_k)
            res2 = tf.reduce_sum(res2) * 2. / (nf * nf)
            stat = res1 - res2
        elif self.kernel == 'IMQ':
            # k(x, y) = C / (C + ||x - y||^2)
            # C = tf.nn.top_k(tf.reshape(distances, [-1]), half_size).values[half_size - 1]
            # C += tf.nn.top_k(tf.reshape(distances_qz, [-1]), half_size).values[half_size - 1]
            #if opts['pz'] == 'normal':
            #    Cbase = 2. * opts['zdim'] * sigma2_p
            #elif opts['pz'] == 'sphere':
            #    Cbase = 2.
            #elif opts['pz'] == 'uniform':
                # E ||x - y||^2 = E[sum (xi - yi)^2]
                #               = zdim E[(xi - yi)^2]
                #               = const * zdim
            #    Cbase = opts['zdim']

            Cbase = 2. * self.config['latent_dim'] * 2. * 1. # sigma2_p # for normal sigma2_p = 1
            stat = 0.
            for scale in [.1, .2, .5, 1., 2., 5., 10.]:
                C = Cbase * scale
                res1 = C / (C + distances_qz)
                res1 += C / (C + distances_pz)
                res1 = tf.multiply(res1, 1. - tf.eye(n))
                res1 = tf.reduce_sum(res1) / (nf * nf - nf)
                res2 = C / (C + distances)
                res2 = tf.reduce_sum(res2) * 2. / (nf * nf)
                stat += res1 - res2
        return stat


    def wae_loss(self, x, z_decoded):

        wass_loss = self.mmd_penalty(self.z_prior, z_decoded)
  
        x = K.flatten(x)
        z_decoded = K.flatten(z_decoded)

        # Reconstruction loss
        xent_loss = keras.metrics.binary_crossentropy(x, z_decoded)
        #for s in self.img_shape:
        #    xent_loss *= s
        # xent_loss *= 224 * 224 * 3

        # KL divergence
        kl_loss = - K.sum(1 + self.z_log_sigma - K.square(self.z_mu) - K.exp(self.z_log_sigma), axis=-1)
  
        wass_loss = tf.identity(wass_loss)
        kl_loss = tf.identity(kl_loss)
        xent_loss = tf.identity(xent_loss)
  
        if self.debug:
            wass_loss = tf.Print(wass_loss, [wass_loss], 'wass_loss: ')
            kl_loss = tf.Print(kl_loss, [kl_loss], 'kl_loss: ')
            xent_loss = tf.Print(xent_loss, [xent_loss], 'reconstruction_loss: ')

        return tf.reduce_mean(self.recon_lambda * xent_loss + \
                      self.kl_lambda * kl_loss + \
                      self.wae_lambda * wass_loss)

    def vae_loss(self, x, z_decoded):
        x = K.flatten(x)
        z_decoded = K.flatten(z_decoded)

        # Reconstruction loss
        xent_loss = keras.metrics.binary_crossentropy(x, z_decoded)
        for s in self.img_shape:
            xent_loss *= s
        # xent_loss *= 224 * 224 * 3

        # KL divergence
        kl_loss = - K.sum(1 + self.z_log_sigma - K.square(self.z_mu) - K.exp(self.z_log_sigma), axis=-1)
        return K.mean(self.recon_lambda * xent_loss + \
                      self.kl_lambda * kl_loss)

    
    def get_ckpointer(self):
        filepath = "checkpoints/weights_vgg_wae.{epoch:03d}-{val_loss:.4f}.ckpt"
        checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=1, 
                                    save_best_only=True, save_weights_only=True,mode='auto', period=1)
        callbacks_list = [checkpoint]
        return callbacks_list

    def save_latent(self):
      encoder = Model(self.input_img, self.z_mu)
      z_mu_new = encoder.predict(self.X_val, batch_size=self.batch_size)


      dd = lambda d, f: pickle.dump(d, open(f, "wb"))
      dd(z_mu_new, "z_mu.p")
      dd(self.y_val, "y_val.p")
      dd(self.artists, "artists.p")
      dd(prior_dists_np, "priors.p")


    def train_supervised(self, epochs=None):
            
        if epochs is None:
            epochs = self.epochs
        
        #model = self.build_model(wae_loss)

        callbacks_list = self.get_ckpointer()

        print("STARTING EPOCHS...")

        def data_gen(X, y):
          n = len(X)
          i = 0

          while i < n:

            yield X[i:i+self.batch_size]/255, y[i:i+self.batch_size]
            i += self.batch_size

        old_vloss = 0
        err = 0.0001
        val_zeros = np.zeros(self.X_val.shape[0])

        for e in range(epochs):
          i = 0
          for X_train, y_train in data_gen(self.X[:self.train_num - (self.train_num % self.batch_size)],
                                  self.y[:self.train_num - (self.train_num % self.batch_size)]):
              i += self.batch_size
              if i > (self.batch_size*10) and i % (self.batch_size*10) == 0:
                  print(i)
              if (i > self.batch_size*100) and (i % (self.batch_size*100) == 0): # every 1k images, save latest
                hist = self.model.fit(x=[X_train, y_train], y=X_train,
                      shuffle=False,
                      epochs=1,
                      callbacks=callbacks_list,
                      batch_size=self.batch_size,
                      validation_data=([self.X_val, val_zeros], self.X_val))
               
                vloss = hist.history['val_loss'][-1]

                if abs(vloss - old_vloss) < err:
                    break
                old_vloss = vloss
              else:
                  self.model.fit(x=[X_train, y_train], y=X_train,
                      shuffle=False,
                      epochs=1,
                verbose=int(i % (self.batch_size*10) == 0),
                      batch_size=self.batch_size)


    def train(self, epochs=None):
            
        if epochs is None:
            epochs = self.epochs
        
        #model = self.build_model(wae_loss)

        callbacks_list = self.get_ckpointer()

        print("STARTING EPOCHS...")

        def data_gen(X):
          n = len(X)
          i = 0

          while i < n:

            yield X[i:i+self.batch_size]/255
            i += self.batch_size

        old_vloss = 0
        err = 0.0001

        for e in range(epochs):
          i = 0
          for X_train in data_gen(self.X[:self.train_num - (self.train_num % self.batch_size)]):
              i += self.batch_size
              if i > (self.batch_size*10) and i % (self.batch_size*10) == 0:
                  print(i)
              if (i > self.batch_size*100) and (i % (self.batch_size*100) == 0): # every 1k images, save latest
                hist = self.model.fit(x=X_train, y=X_train,
                      shuffle=False,
                      epochs=1,
                      callbacks=callbacks_list,
                      batch_size=self.batch_size,
                      validation_data=(self.X_val, self.X_val))
               
                vloss = hist.history['val_loss'][-1]

                if abs(vloss - old_vloss) < err:
                    break
                old_vloss = vloss
              else:
                  self.model.fit(x=X_train, y=X_train,
                      shuffle=False,
                      epochs=1,
                verbose=int(i % (self.batch_size*10) == 0),
                      batch_size=self.batch_size)



model = CNN_VAE()
model.build_model(model.wae_loss)
model.train_supervised(epochs=0)
model.save_latent()


