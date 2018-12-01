import keras
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Flatten, Input
from keras.layers import Conv2D, MaxPooling2D, UpSampling2D,Deconv2D
from keras import backend as K
import skimage
import numpy as np
import os
import sys
import matplotlib.image as mpimg
from encoder import Encoder

class Autoencoder:
    def __init__(self, filters=[60,30], kernel_size=(10,10), poll_size=(6,6), input_size=(360,360,1), epochs = 5, batch_size= 16):
        self.encoder = Encoder(filters=filters, kernel_size=kernel_size, poll_size=poll_size, input_size=input_size)
        self.decoder = Decoder(filters=filters.reverse(), kernel=kernel_size, poll=poll_size, input_size= self.encoder.get_out_length())
        
        self.input_size = input_size
        self.network = None
        self.train_data = None
        self.test_data = None
        self.weights = None
        self.epochs = epochs
        self.batch_size = batch_size
                 
                 
    def createNetwork():
        autoencoder = Model(Input(shape = self.input_size), self.decoder(self.encoder(Input(shape = self.input_size)))
        autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy')
        self.network = autoencoder
        
        
    def getData(directory = '',data_size=2000, train_ratio = 0.8):
        if (directory is None):
            print('Data not present')
            return
            
        train_size = int(train_ratio * data_size)
        test_size = data_size - train_size
        
        train_data=np.zeros(shape=tup([train_size])+self.input_size)
        test_data=np.zeros(shape=tup([test_size])+self.input_size)
        idx = 0

        for f in os.listdir(directory):
            if(idx == data_size):
                break
            if(idx%5 == 0):
                if(idx < train_size):
                    train_data[idx,:,:,:] = train_data[idx,:,:,:] + np.reshape(skimage.color.rgb2gray(mpimg.imread(directory+f)),self.input_size)
                else:
                    test_data[idx-train_size,:,:,:] = test_data[idx-train_size,:,:,:] + np.reshape(skimage.color.rgb2gray(mpimg.imread(directory+f)),self.input_size)
            idx = idx +1
            
        self.train_data = train_data
        self.test_data = test_data
        
        
    def train(directory = ''):
        print('Training about to start!')
        self.network.fit(self.train_data, self.train_data, epochs=self.epochs, batch_size=self.batch_size, shuffle=True, validation_data=(self.test_data, self.test_data), verbose=1)
        self.network.save_weights('autoencoder_weights.h5f')
        self.network.encoder.network.save_weights('encoder_weights.h5f')
        self.network.decoder.network.save_weights('decoder_weights.h5f')
        self.weights = self.network.get_weights()
                   
