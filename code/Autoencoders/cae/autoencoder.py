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
from decoder import Decoder

class Autoencoder:
    def __init__(self, filters=[60,30], kernel_size=(10,10), poll_size=(6,6), input_size=(360,360,1), epochs = 5, batch_size= 16):
        self.encoder = Encoder(filters=filters, kernel_size=kernel_size, poll_size=poll_size, input_size=input_size)
        self.encoder.createNetwork()
        self.decoder = Decoder(filters=filters[::-1], kernel_size=kernel_size, poll_size=poll_size, input_size= self.encoder.get_out_length())
        self.decoder.createNetwork()
        self.input_size = input_size
        self.network = None
        self.train_data = None
        self.test_data = None
        self.weights = None
        self.epochs = epochs
        self.batch_size = batch_size
                 
                 
    def createNetwork(self):
        input_img = Input(shape = self.input_size)
        autoencoder = Model(input_img, self.decoder.network(self.encoder.network(input_img)))
        autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy')
        self.network = autoencoder
        print('Autoencoder Network created Successfully!')
        
        
    def getData(self,directory = '',data_size=2000, train_ratio = 0.8):
        if (directory is None):
            print('Data not present')
            return
            
        train_size = int(train_ratio * data_size)
        test_size = data_size - train_size
        
        train_data=np.zeros(shape=tuple([train_size])+self.input_size)
        test_data=np.zeros(shape=tuple([test_size])+self.input_size)
        idx = 0
        print('Fetching Data!')
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
        print('Data acquisition successfull')
        
    def train(self,directory = ''):
        print('Training about to start!')
        self.network.fit(self.train_data, self.train_data, epochs=self.epochs, batch_size=self.batch_size, shuffle=True, validation_data=(self.test_data, self.test_data), verbose=1)
        self.network.save_weights(directory+'autoencoder_weights.h5f')
        self.encoder.network.save_weights(directory+'encoder_weights.h5f')
        self.decoder.network.save_weights(directory+'decoder_weights.h5f')
        self.weights = self.network.get_weights()
        
        
    def load_weights(self,directory = ''):
        self.network.load_weights('autoencoder_weights.h5f')
        self.encoder.network.load_weights('encoder_weights.h5f')
        self.decoder.network.load_weights('decoder_weights.h5f')
        
        print('Weights loaded Successfully!')
        
        
    def predict(self, file_dir = ''):
        img = np.reshape(skimage.color.rgb2gray(mpimg.imread(file_dir)),(1,360,360,1))
        enc_img = self.encoder.network.predict(img)
        dec_img = self.decoder.network.predict(enc_img)
        autoenc_img = self.network.predict(img)
        
        mpimg.imsave('grey_image.png',np.reshape(img,(360,360)))
        mpimg.imsave('decoded_image.png',np.reshape(dec_img,(360,360)))
        mpimg.imsave('autoencoder_image.png',np.reshape(autoenc_img,(360,360)))
        for i in np.arange(30):
            mpimg.imsave('enc'+str(i)+'_img.png',enc_img[0,:,:,i])
            
        print('Prediction Completed Successfully!')
        
