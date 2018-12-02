import numpy as np
import math
from keras.models import model_from_json, load_model
from keras.models import Sequential
from keras.layers import Dense, Flatten, Input, Add, Lambda, Activation, concatenate
from keras.models import Sequential, Model
from keras.optimizers import Adam
import keras.backend as K
import tensorflow as tf

HIDDEN1_UNITS = 600
HIDDEN2_UNITS = 600

class CriticNetwork(object):

    def __init__(self, sess, S, I, action_size, n_oth_var, BATCH_SIZE, TAU, LEARNING_RATE):
    
        self.sess = sess
        self.BATCH_SIZE = BATCH_SIZE
        self.TAU = TAU
        self.LEARNING_RATE = LEARNING_RATE
        self.action_size = action_size
        
        K.set_session(sess)

        #Now create the model
        self.model, self.action, self.state_img, self.state_oth = self.create_critic_network(S, I, action_size, n_oth_var )  
        self.target_model, self.target_action, self.target_state_img, self.target_state_oth = self.create_critic_network(S, I, action_size, n_oth_var )  
        self.action_grads = tf.gradients(self.model.output, self.action)  #GRADIENTS for policy update
        self.sess.run(tf.initialize_all_variables())

    def gradients(self, states_img, states_oth, actions):
    
        return self.sess.run(self.action_grads, feed_dict={
            self.state_img: states_img, self.state_oth: states_oth, self.action: actions
        })[0]

    def target_train(self):
    
        critic_weights = self.model.get_weights()
        critic_target_weights = self.target_model.get_weights()
    
        for i in range(len(critic_weights)):
            critic_target_weights[i] = self.TAU * critic_weights[i] + (1 - self.TAU)* critic_target_weights[i]
    
        self.target_model.set_weights(critic_target_weights)

    def create_critic_network(self, S, I, action_dim, n_oth_var):
    
        print("Now we build the model")
        #S = Input(shape=[state_size])  
        other_variables = Input(shape=[n_oth_var])
        state_vector = concatenate([S, other_variables],axis=-1)
    
        A = Input(shape=[action_dim],name='action2')   
        w1 = Dense(HIDDEN1_UNITS, activation='relu')(state_vector)
        a1 = Dense(HIDDEN2_UNITS, activation='linear')(A) 
        h1 = Dense(HIDDEN2_UNITS, activation='linear')(w1)
    
        h2 = Add()([h1, a1])
    
        h3 = Dense(HIDDEN2_UNITS, activation='relu')(h2)
        V = Dense(action_dim,activation='linear')(h3)   
        model = Model(inputs=[I, other_variables, A],outputs=V)
        adam = Adam(lr=self.LEARNING_RATE)
        model.compile(loss='mse', optimizer=adam)
    
        return model, A, I, other_variables
