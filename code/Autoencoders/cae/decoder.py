import keras
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Flatten, Input
from keras.layers import Conv2D, MaxPooling2D, UpSampling2D,Deconv2D
from keras import backend as K
import numpy as np

class Decoder(object):
    def __init__(self, filters=[30,60], kernel_size=(10,10), poll_size=(6,6), input_size=None):
        #self.layers = layers
        self.filters = filters
        self.kernel_size = kernel_size
        self.poll_size = poll_size
        self.input_size = input_size
        self.network = None
        self.train_data = None
        self.test_data = None
        if (weights is not None):
            self.weights = keras.load_weights(weights)
        else:
            self.weights = None
        
        
    def createNetwork(self):
        input_img = Input(shape = self.input_size)
        x = Conv2D(self.filters[0], kernel_size=self.kernel_size, activation='relu', border_mode='same')(input_img) #nb_filter, nb_row, nb_col
        x = UpSampling2D(pool_size=self.poll_size)(x)
        x = Conv2D(self.filters[1], kernel_size=self.kernel_size, activation='relu', border_mode='same')(x)
        x = UpSampling2D(pool_size=self.poll_size)(x)
        decoded = Conv2D(1, kernel_size=kernel_tup, activation='sigmoid', border_mode='same')(x)
        self.network = Model(input_img, decoded)
        print('Network created successfully!')
        
        
    def load_weights(self,filename = ''):
        self.network.load_weights(filename)
        self.weights = self.network.get_weights()
        

    def predict(self,image):
        return self.network.predict(image)
