import sys 
from os import environ
sys.path.append(environ["CARLA_ROOT"]+'/Setting-Up-CARLA-RL')
from Environment.carla_environment_wrapper import CarlaEnvironmentWrapper as CarlaEnv
import numpy as np
import random
import argparse
from keras.models import model_from_json, Model
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers import Input, concatenate
from keras.optimizers import Adam
import tensorflow as tf
import json
from threading import Thread
import time
from ReplayBuffer import ReplayBuffer
from ActorNetwork import ActorNetwork
from CriticNetwork import CriticNetwork
from preprocessing import PreProcessing
from OU_noise import OU_noise
import time
import matplotlib.pyplot as plt 
import cv2



noise = OU_noise()       #noise based on Ornstein-Uhlenbeck process

def Simulate_Game(train_indicator=1):    #1 means Train, 0 means simply Run

    BUFFER_SIZE = 100000
    BATCH_SIZE = 32
    GAMMA = 0.99

    TAU = 0.001     #Target Network HyperParameters
    LRA = 0.0001    #Learning rate for Actor
    LRC = 0.001     #Lerning rate for Critic

    action_dim = 3  #Steering/Acceleration/Brake
    state_dim = 6  #of sensors input

    np.random.seed(1337)

    vision = False

    EXPLORE = 100000.
    episode_count = 2000
    max_steps = 100000
    reward = 0
    done = False
    step = 0
    epsilon = 1
    indicator = 0
    n_other_var = 6

    #Tensorflow GPU optimization
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    from keras import backend as K
    K.set_session(sess)

    preprocessor = PreProcessing(sess, n_other_var)
    S, I = preprocessor.build_CNN_model()

    actor = ActorNetwork(sess, S, I, action_dim, n_other_var, BATCH_SIZE, TAU, LRA) 
    critic = CriticNetwork(sess, S, I, action_dim, n_other_var, BATCH_SIZE, TAU, LRC)
    buff = ReplayBuffer(BUFFER_SIZE)    #Create replay buffer
    
    
    print("Creating Environment..")
    env = CarlaEnv(is_render_enabled=False, num_speedup_steps = 10, run_offscreen=False, cameras = ['SceneFinal', 'Depth', 'SemanticSegmentation'], save_screens=True)

    #Now load the weight
    print("Now we load the weight")
    
    try:
        actor.model.load_weights("actormodel.h5")
        critic.model.load_weights("criticmodel.h5")
        actor.target_model.load_weights("actormodel.h5")
        critic.target_model.load_weights("criticmodel.h5")
        print("Weight load successfully")

    except:
        print("Cannot find the weight")
    
    print("CARLA Experiment Start.")




    for i in range(episode_count):

        print("Episode : " + str(i) + " Replay Buffer " + str(buff.count()))
        

        ob = env.reset()
        
        s_t = np.hstack((preprocessor.image_preprocess(ob['rgb_image']), ob['acceleration'].x, ob['acceleration'].y, ob['forward_speed'], ob['rotation'], ob['intersection_otherlane'], ob['intersection_offroad']))
        
        total_reward = 0.

        for j in range(max_steps):
        
            loss = 0 
            epsilon -= 1.0 / EXPLORE
            a_t = np.zeros([1,action_dim])
            noise_t = np.zeros([1,action_dim])
            
            # Extract image and other data out of the latest experience tuple from the stack
            image_t, n_other_state_t = preprocessor.state_to_img_n_other_var_nn(s_t)
            a_t_original = actor.model.predict([image_t, n_other_state_t])

            noise_t[0][0] = train_indicator * max(epsilon, 0) * noise.function(a_t_original[0][0],  0.5 , 1, 0.30)
            noise_t[0][1] = train_indicator * max(epsilon, 0) * noise.function(a_t_original[0][1],  0 , 1.0, 0.550)
            noise_t[0][2] = train_indicator * max(epsilon, 0) * noise.function(a_t_original[0][2], 0.05 , 0.600, 0.05)

            #The following code do the stochastic brake
            #if random.random() <= 0.1:
            #    print("********Now we apply the brake***********")
            #    noise_t[0][2] = train_indicator * max(epsilon, 0) * noise.function(a_t_original[0][2],  0.2 , 1.00, 0.10)

            a_t[0][0] = a_t_original[0][0] + noise_t[0][0]
            a_t[0][1] = a_t_original[0][1] + noise_t[0][1]
            a_t[0][2] = a_t_original[0][2] + noise_t[0][2]

            ob, r_t, done, info = env.step(a_t[0])
            
            s_t1 = np.hstack((preprocessor.image_preprocess(ob['rgb_image']), ob['acceleration'].x, ob['acceleration'].y, ob['forward_speed'], ob['rotation'], ob['intersection_otherlane'], ob['intersection_offroad']))
            
            buff.add(s_t, a_t[0], r_t, s_t1, done)      #Add replay buffer

            #Do the batch update
            batch = buff.getBatch(BATCH_SIZE)
            states_img = np.asarray([preprocessor.state_to_img_n_other_var_nn(e[0])[0][0] for e in batch])
            states_oth = np.asarray([preprocessor.state_to_img_n_other_var_nn(e[0])[1][0] for e in batch])
            
            actions = np.asarray([e[1] for e in batch])
            rewards = np.asarray([e[2] for e in batch])
            new_states_img = np.asarray([preprocessor.state_to_img_n_other_var_nn(e[3])[0][0] for e in batch])
            new_states_oth = np.asarray([preprocessor.state_to_img_n_other_var_nn(e[3])[1][0] for e in batch])
            
            dones = np.asarray([e[4] for e in batch])
            y_t = np.asarray([e[1] for e in batch])

            target_q_values = critic.target_model.predict([new_states_img, new_states_oth, actor.target_model.predict([new_states_img, new_states_oth])])  
           
            if j < BATCH_SIZE:
                l = j
            else:
                l = BATCH_SIZE

            for k in range(l):

                if dones[k]:
                    y_t[k] = rewards[k]

                else:
                    y_t[k] = rewards[k] + GAMMA*target_q_values[k]
      
            if (train_indicator):
                loss += critic.model.train_on_batch([states_img, states_oth, actions], y_t) 
                a_for_grad = actor.model.predict([states_img, states_oth])
                grads = critic.gradients(states_img, states_oth, a_for_grad)

                actor.train(states_img, states_oth, grads)
                actor.target_train()
                critic.target_train()

            total_reward += r_t
            s_t = s_t1
        
            #print("Episode", i, "Step", step, "Action", a_t, "Reward", r_t, "Loss", loss)
            #env.save_screenshots()

            step += 1
            if done:
                break
             
        if np.mod(i, 10) == 0:
            if (train_indicator):
                print("Now we save model")
                actor.model.save_weights("actormodel.h5", overwrite=True)
                with open("actormodel.json", "w") as outfile:
                    json.dump(actor.model.to_json(), outfile)

                critic.model.save_weights("criticmodel.h5", overwrite=True)
                with open("criticmodel.json", "w") as outfile:
                    json.dump(critic.model.to_json(), outfile)
        
        print("TOTAL REWARD @ " + str(i) +"-th Episode  : Reward " + str(total_reward))
        print("Total Step: " + str(step))
        print("")

    env.end()  # This is for shutting down TORCS
    print("Finish.")
    
if __name__ == "__main__":
 
    Simulate_Game()
    