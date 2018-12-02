import keras
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Flatten, Input
from keras.layers import Conv2D, MaxPooling2D, UpSampling2D,Deconv2D
from keras import backend as K
import numpy as np

class Encoder(object):
    def __init__(self, filters=[60,30], kernel_size=[(10,10)], poll_size=(6,6), input_size=(360,360,1)):
        #self.layers = layers
        self.filters = filters
        self.kernel_size = kernel_size
        self.poll_size = poll_size
        self.input_size = input_size
        self.network = None
        self.train_data = None
        self.test_data = None
        self.output_size = None
        self.weights = None
        
        
    def createNetwork(self):
        
        input_img = Input(shape = self.input_size)
        x = Conv2D(self.filters[0], kernel_size=self.kernel_size[0], activation='relu', border_mode='same')(input_img) #nb_filter, nb_row, nb_col
        x = MaxPooling2D(pool_size=self.poll_size, border_mode='same')(x)
        x = Conv2D(self.filters[1], kernel_size=self.kernel_size[1], activation='relu', border_mode='same')(x) #nb_filter, nb_row, nb_col
        x = MaxPooling2D(pool_size=self.poll_size, border_mode='same')(x)
        x = Conv2D(self.filters[2], kernel_size=self.kernel_size[2], activation='relu', border_mode='same')(x)
        encoded = MaxPooling2D(pool_size=self.poll_size, border_mode='same')(x)
        self.output_size = K.int_shape(encoded)[1:]
        self.network = Model(input_img, encoded)
        print('Encoder Network created successfully!')
        
        
    def load_weights(self,filename = ''):
        self.network.load_weights(filename)
        self.weights = self.network.get_weights()
        

    def predict(self,image):
        return self.network.predict(image)
        
    
    def get_out_length(self):
        return self.output_size
    
        
