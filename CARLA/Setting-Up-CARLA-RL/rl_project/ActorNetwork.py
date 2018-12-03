import numpy as np
import math
from keras.initializers import VarianceScaling
from keras.models import model_from_json
from keras.models import Sequential, Model
from keras.layers import Dense, Flatten, Input, concatenate, Lambda
from keras.optimizers import Adam
import tensorflow as tf
import keras.backend as K

HIDDEN1_UNITS = 250
HIDDEN2_UNITS = 250

class ActorNetwork(object):

    def __init__(self, sess, S, I1, I2, I3, I4, action_size, n_oth_var, BATCH_SIZE, TAU, LEARNING_RATE, n_frames):

        self.sess = sess
        self.BATCH_SIZE = BATCH_SIZE
        self.TAU = TAU
        self.LEARNING_RATE = LEARNING_RATE

        K.set_session(sess)

        #Now create the model
        self.model , self.weights, self.state_img1, self.state_img2, self.state_img3, self.state_img4, self.state_oth = self.create_actor_network(S, I1, I2, I3, I4, action_size, n_oth_var, n_frames)   
        self.target_model, self.target_weights, self.target_state_img1, self.target_state_img2, self.target_state_img3, self.target_state_img4, self.target_state_oth = self.create_actor_network(S, I1, I2, I3, I4, action_size,n_oth_var, n_frames) 
        self.action_gradient = tf.placeholder(tf.float32,[None, action_size])
        self.params_grad = tf.gradients(self.model.output, self.weights, -self.action_gradient)
        grads = zip(self.params_grad, self.weights)
        self.optimize = tf.train.AdamOptimizer(LEARNING_RATE).apply_gradients(grads)
        self.sess.run(tf.initialize_all_variables())

    def train(self, states_img1, states_img2, states_img3, states_img4, states_oth, action_grads):
        self.sess.run(self.optimize, feed_dict={
            self.state_img1: states_img1, self.state_img2: states_img2, self.state_img3: states_img3, self.state_img4: states_img4,
            self.state_oth: states_oth,
            self.action_gradient: action_grads
        })

    def target_train(self):
     
        actor_weights = self.model.get_weights()
        actor_target_weights = self.target_model.get_weights()
     
        for i in range(len(actor_weights)):
            actor_target_weights[i] = self.TAU * actor_weights[i] + (1 - self.TAU)* actor_target_weights[i]
     
        self.target_model.set_weights(actor_target_weights)

    def create_actor_network(self, S, I1, I2, I3, I4, action_dim, n_oth_var, n_frames):

        print("Now, we build the actor network model")
        
        #S = Input(shape=[state_size])   
        other_variables = Input(shape=[n_oth_var*n_frames])
        state_vector = concatenate([S, other_variables],axis=-1)
    
        h0 = Dense(HIDDEN1_UNITS, activation='relu')(state_vector)
        h1 = Dense(HIDDEN2_UNITS, activation='relu')(h0)
        
        Steering = Dense(1,activation='tanh',kernel_initializer=lambda shape:VarianceScaling(scale=1e-4)(shape))(h1) #Dense(1,activation='tanh',init=lambda shape, name: normal(shape, scale=1e-4, name=name))(h1)  
        Acceleration = Dense(1,activation='sigmoid',kernel_initializer=lambda shape:VarianceScaling(scale=1e-4)(shape))(h1)#Dense(1,activation='sigmoid',init=lambda shape, name: normal(shape, scale=1e-4, name=name))(h1)   
        Brake = Dense(1,activation='sigmoid',kernel_initializer=lambda shape:VarianceScaling(scale=1e-4)(shape))(h1) #Dense(1,activation='sigmoid',init=lambda shape, name: normal(shape, scale=1e-4, name=name))(h1) 
        
        V = concatenate([Steering, Acceleration, Brake])
        model = Model(inputs=[I1, I2, I3, I4,other_variables], outputs=V)
        
        return model, model.trainable_weights, I1, I2, I3, I4, other_variables

