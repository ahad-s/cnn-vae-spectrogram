#!/usr/bin/env python
# coding: utf-8

r_seed = 12345
GPU = True
import os

if GPU:

  os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
  os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import pandas as pd
import numpy as np
# np.random.seed(r_seed)
import tensorflow as tf
# tf.set_random_seed(r_seed)
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
    def __init__(self, n_images=-1):
      self.img_shape = (224, 224, 3)
      self.batch_size = 32
      self.latent_dim = 50  # Number of latent dimension parameters
      self.epochs = 1
      self.artists = {}
      self.artist_num = 10 # num_classes
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
      #self.weight_file = 'checkpoints/weights_vgg_wae.001-6.0998.ckpt'
      self.model = None 

      self.sp_folder = "spectrogram_images/"
      self.sp_folder = "/collection/aghabuss/datasets/spectrogram_images/"
      self.populate_images(n_images)
        

#       self.get_z_prior_samples()

#       self.labels = 

      self.prior_means = []
      self.prior_sigmas = []
      self.prior_dists = []
      self.prior_dists_np = []

      self.input_labels = keras.Input(shape=(1,), dtype='int32')

      self.z_prior = self.get_z_prior_samples()


    def gumbel_softmax(self, weights, hard=True):
      epsilon = 1e-20

      log_weights = tf.log(weights)

      u = tf.random_uniform(tf.shape(log_weights), minval=0, maxval=1)
      gmbl_noise = -tf.log(-tf.log(u + epsilon) + epsilon)

      y = tf.nn.softmax(log_weights + gmbl_noise)

      if hard:
        k = tf.shape(log_weights)[-1]
        y_hard = tf.cast(tf.equal(y,tf.reduce_max(y, axis=-1)), y.dtype)
        y = tf.stop_gradient(y_hard - y) + y

      return y


    def get_z_prior(self):

      epsilon = K.random_normal(shape=(self.batch_size, self.latent_dim),
                                mean=0., stddev=1.)
      self.z_prior = self.prior_mu + self.prior_sigma * epsilon
      return self.z_prior


    def get_z_prior_samples(self):
      a, b, = -2, 2 # 2 stddevs away from N(0,1), as in tf.trunc_norm_initializer
      self.prior_weights = [] 

      for i in range(self.artist_num):
        epsilon = K.random_normal([self.batch_size, self.latent_dim])
        mean = truncnorm(a, b).rvs(self.latent_dim)
        sigma = truncnorm(a, b).rvs(self.latent_dim)

        self.prior_dists_np.append((mean, sigma))

        mean = tf.convert_to_tensor(mean, name="prior_mean_{}".format(str(i)), dtype=tf.float32)
        sigma = tf.convert_to_tensor(sigma, name="prior_sigma_{}".format(str(i)), dtype=tf.float32)
        weight = tf.Variable(-(i+1), name="prior_weight_{}".format(str(i)), dtype=tf.float32)
        dist = K.cast(mean + epsilon * K.exp(sigma), dtype='float32')

        self.prior_means.append(mean)
        self.prior_sigmas.append(sigma)
        self.prior_weights.append(weight)
        self.prior_dists.append(dist)
    
      self.z_prior = tf.zeros(shape=(self.batch_size, self.latent_dim))
      self.alphas = self.gumbel_softmax(self.prior_weights, hard=True)
      for i in range(self.artist_num):
        alpha_i = tf.nn.embedding_lookup(self.alphas, [i])
        print(alpha_i)
        print(self.prior_dists[i])
        break
        self.z_prior += alpha_i * self.prior_dists[i]

      print("---Z PRIOR---")
      print(self.z_prior)
      #self.z_prior = tf.reshape(self.z_prior, [-1, self.latent_dim])
      return self.z_prior



    def populate_images(self):
        Z_train = pd.read_csv("train.csv")
      # y_train = Z_train['label'].values
        X = Z_train.drop('label', axis=1).values

        self.X = X.reshape(X.shape[0], 28, 28, 1)
        self.train_num = int((1-self.val_pct)*len(self.X))

        self.X_val = X[self.train_num:] / 255
        self.y_val = y[self.train_num:]
        
        self.X_train = self.X[:self.train_num]
        self.y_train = self.y[:self.train_num]

        # indices of each artist x_i-y_i
        index_sets = [[] for i in range(10)]
        for i in range(len(self.y_train)):
          index_sets[self.y_train[i]].append(i)

        # should sum up to 1
        self.artist_probs = [0.1] * 10

        self.X_splits = []
        self.y_splits = []
        for idxs in index_sets:
          self.X_splits.append(self.X_train[idxs])
          self.y_splits.append(self.y_train[idxs])


    def encoder(self, input_img):
        # Encoder architecture: VGG16 encoder with VGG16-style DCNN decoder

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

        self.shape_before_flattening = K.int_shape(x)

        x = Flatten()(x)

        x = Dense(32, activation='relu')(x)
#         x = Dense(32, activation='relu')(x)

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

        # Expand to 224x224 total pixel
        # use Conv2DTranspose to reverse the conv layers from the encoder
        # Expand to 784 total pixels
        x = layers.Dense(np.prod(self.shape_before_flattening[1:]),
                         activation='relu')(decoder_input)

        # reshape
        x = layers.Reshape(self.shape_before_flattening[1:])(x)        

        x = layers.Conv2DTranspose(32, 3,
                                   padding='same', 
                                   activation='relu',
                                   strides=(2, 2))(decoder_input)
        x = layers.Conv2D(1, 3,
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
        print(self.z_prior, self.z)
        print("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@")
        print("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@")
        print("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@")
        wass_loss = self.mmd_penalty(self.z_prior, self.z)
  
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
        filepath = "checkpoints/unsup_weights_vgg_wae.{epoch:03d}-{val_loss:.4f}.ckpt"
        checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=1, 
                                    save_best_only=False, save_weights_only=True,mode='auto', period=1)
        callbacks_list = [checkpoint]
        return callbacks_list

    def save_latent(self):
      encoder = Model(self.input_img, self.z_mu)
      z_mu_new = encoder.predict(self.X_val, batch_size=self.batch_size)


      dd = lambda d, f: pickle.dump(d, open(f, "wb"))
      dd(z_mu_new, "z_mu_unsup.p")
      dd(self.y_val, "y_val_unsup.p")
      dd(self.artists, "artists_unsup.p")
      dd(self.prior_dists_np, "priors_unsup.p")


    def train_unsupervised(self, epochs=None):
            
        if epochs is None:
            epochs = self.epochs

        callbacks_list = self.get_ckpointer()

        print("STARTING EPOCHS...")

        weight_probs = []


        """
        Each batch is of the same label/artist
        """
        def data_gen(X, y):
          n = len(X)
          i = 0
          num_batches = n // self.batch_size
          labels = list(range(self.artist_num))

          for i in range(num_batches):
            rnd_lab = np.random.choice(labels, p=self.artist_probs)
            X_in_batch = self.X_splits[rnd_lab]
            y_in_batch = self.y_splits[rnd_lab]
            batch_idx = np.random.choice(np.arange(len(X_in_batch)), 
                                          self.batch_size,
                                          replace=False)
            yield X_in_batch[batch_idx] / 255, y_in_batch[batch_idx]

        """
        def data_gen(X, y):
          n = len(X)
          i = 0

          while i < n:

            yield X[i:i+self.batch_size]/255, y[i:i+self.batch_size]
            i += self.batch_size
        """

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




model = CNN_VAE()
model.build_model(model.wae_loss)
try:
    model.train_supervised(epochs=0)
    model.save_latent()
except KeyboardInterrupt:
    model.save_latent()


