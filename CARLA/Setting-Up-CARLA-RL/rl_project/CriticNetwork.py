import numpy as np
import math
from keras.models import model_from_json, load_model
from keras.models import Sequential
from keras.layers import Dense, Flatten, Input, Add, Lambda, Activation, concatenate
from keras.models import Sequential, Model
from keras.optimizers import Adam
import keras.backend as K
import tensorflow as tf

HIDDEN1_UNITS = 250
HIDDEN2_UNITS = 250

class CriticNetwork(object):

    def __init__(self, sess, S, I1, I2, I3, I4, action_size, n_oth_var, BATCH_SIZE, TAU, LEARNING_RATE, n_frames):
    
        self.sess = sess
        self.BATCH_SIZE = BATCH_SIZE
        self.TAU = TAU
        self.LEARNING_RATE = LEARNING_RATE
        self.action_size = action_size
        
        K.set_session(sess)

        #Now create the model
        self.model, self.action, self.state_img1, self.state_img2, self.state_img3, self.state_img4, self.state_oth = self.create_critic_network(S, I1, I2, I3, I4, action_size, n_oth_var, n_frames)  
        self.target_model, self.target_action, self.target_state_img1, self.target_state_img2, self.target_state_img3, self.target_state_img4, self.target_state_oth = self.create_critic_network(S, I1, I2, I3, I4, action_size, n_oth_var, n_frames)  
        self.action_grads = tf.gradients(self.model.output, self.action)  #GRADIENTS for policy update
        self.sess.run(tf.initialize_all_variables())

    def gradients(self, states_img1, states_img2, states_img3, states_img4, states_oth, actions):
    
        return self.sess.run(self.action_grads, feed_dict={
            self.state_img1: states_img1, self.state_img2: states_img2, self.state_img3: states_img3, self.state_img4: states_img4, self.state_oth: states_oth, self.action: actions
        })[0]

    def target_train(self):
    
        critic_weights = self.model.get_weights()
        critic_target_weights = self.target_model.get_weights()
    
        for i in range(len(critic_weights)):
            critic_target_weights[i] = self.TAU * critic_weights[i] + (1 - self.TAU)* critic_target_weights[i]
    
        self.target_model.set_weights(critic_target_weights)

    def create_critic_network(self, S, I1, I2, I3, I4, action_dim, n_oth_var, n_frames):
    
        print("Now we build the model")
        #S = Input(shape=[state_size])  
        other_variables = Input(shape=[n_oth_var*n_frames])
        state_vector = concatenate([S, other_variables],axis=-1)
    
        A = Input(shape=[action_dim],name='action2')   
        w1 = Dense(HIDDEN1_UNITS, activation='relu')(state_vector)
        a1 = Dense(HIDDEN2_UNITS, activation='linear')(A) 
        h1 = Dense(HIDDEN2_UNITS, activation='linear')(w1)
    
        h2 = Add()([h1, a1])
    
        h3 = Dense(HIDDEN2_UNITS, activation='relu')(h2)
        V = Dense(1,activation='linear')(h3)   
        model = Model(inputs=[I1, I2, I3, I4, other_variables, A],outputs=V)
        adam = Adam(lr=self.LEARNING_RATE)
        model.compile(loss='mse', optimizer=adam)
    
        return model, A, I1, I2, I3, I4, other_variables
