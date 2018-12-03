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
from Image_data_processing import ImageDataProcessing
from OU_noise import OU_noise
import time
import matplotlib.pyplot as plt 
import cv2
from PrioritizedExperienceReplay.proportional import Experience
from collections import deque

noise = OU_noise()       #noise based on Ornstein-Uhlenbeck process

def Simulate_Game(train_indicator=1):    #1 means Train, 0 means simply Run

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
	n_other_var = 8
	n_frames = 4
	BUFFER_SIZE = n_frames
	
	ALPHA = 0.6 # Prioritized experience replay parameter
	BETA = 0.6
	small_epsilon = .01

	#Tensorflow GPU optimization
	config = tf.ConfigProto()
	config.gpu_options.allow_growth = True
	sess = tf.Session(config=config)
	from keras import backend as K
	K.set_session(sess)

	preprocessor = ImageDataProcessing(sess, n_other_var, n_frames)
	S, I1, I2, I3, I4 = preprocessor.build_CNN_model()

	actor = ActorNetwork(sess, S, I1, I2, I3, I4, action_dim, n_other_var, BATCH_SIZE, TAU, LRA, n_frames) 
	critic = CriticNetwork(sess, S, I1, I2, I3, I4, action_dim, n_other_var, BATCH_SIZE, TAU, LRC, n_frames)
	buff = ReplayBuffer(BUFFER_SIZE)    #Create replay buffer
	Exp = Experience(max_steps, ALPHA)
	
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

		#print("Episode : " + str(i) + " Replay Buffer " + str(buff.count()))
		
		BETA = BETA + 0.4*(i/episode_count)
		ob = env.reset()
		s_t = (np.array(ob['rgb_image']), ob['acceleration'].x, ob['acceleration'].y, ob['forward_speed'], ob['location'].x, ob['location'].y, ob['rotation'], ob['intersection_otherlane'], ob['intersection_offroad'])
		S_t = list(zip(s_t, s_t, s_t, s_t))
		R_t = 0
		total_reward = 0.
		s_t1 = []

		for j in range(max_steps):
			  
			loss = 0 
			epsilon -= 1.0 / EXPLORE
			a_t = np.zeros([1,action_dim])
			noise_t = np.zeros([1,action_dim])
			k_frame = j%n_frames
			
			# Extract image and other data out of the latest experience tuple from the stack
			image1_t, image2_t, image3_t, image4_t, n_other_state_t = preprocessor.state_to_img_n_other_var_nn(S_t)
			a_t_original = actor.model.predict([image1_t, image2_t, image3_t, image4_t, n_other_state_t])

			noise_t[0][0] = train_indicator * max(epsilon, 0) * noise.function(a_t_original[0][0],  0.5 , 1, 0.40)
			noise_t[0][1] = train_indicator * max(epsilon, 0) * noise.function(a_t_original[0][1],  0 , 1.0, 0.550)
			noise_t[0][2] = train_indicator * max(epsilon, 0) * noise.function(a_t_original[0][2], 0.03 , 1.00, 0.05)

			#The following code do the stochastic brake
			#if random.random() <= 0.1:
			#    print("********Now we apply the brake***********")
			#    noise_t[0][2] = train_indicator * max(epsilon, 0) * noise.function(a_t_original[0][2],  0.2 , 1.00, 0.10)

			a_t[0][0] = a_t_original[0][0] + noise_t[0][0]
			a_t[0][1] = a_t_original[0][1] + noise_t[0][1]
			a_t[0][2] = a_t_original[0][2] + noise_t[0][2]

			ob, r_t, done, info = env.step(a_t[0])
			
			s_t1.append(np.array((np.array(ob['rgb_image']), ob['acceleration'].x, ob['acceleration'].y, ob['forward_speed'], ob['location'].x, ob['location'].y, ob['rotation'], ob['intersection_otherlane'], ob['intersection_offroad'])))
			
			R_t = R_t + (GAMMA**k_frame)*r_t

			#buff.add(s_t, a_t[0], r_t, s_t1, done)      #Add to buffer
			if np.mod(j+1, n_frames) == 0:
				#print(len(s_t1))
				S_t1 = list(zip(s_t1[0],s_t1[1],s_t1[2],s_t1[3]))

				Exp.add((S_t, a_t[0], R_t, S_t1, done))
				
				if j < BATCH_SIZE*(n_frames+1):
					continue
				#batch = buff.getBatch(BATCH_SIZE)
				batch, weights, indices = Exp.select(BATCH_SIZE,BETA)
				#print(batch, j)
				states_img = np.asarray([np.array(e[0][0]) for e in batch])
				states_img1 = states_img[:,0,:,:,np.newaxis]
				states_img2 = states_img[:,1,:,:,np.newaxis]
				states_img3 = states_img[:,2,:,:,np.newaxis]
				states_img4 = states_img[:,3,:,:,np.newaxis]
				states_oth = np.reshape(np.asarray([e[0][1:] for e in batch]),(BATCH_SIZE, n_frames*n_other_var,))
				
				actions = np.asarray([e[1] for e in batch])
				rewards = np.asarray([e[2] for e in batch])
				new_states_img = np.asarray([np.array(e[3][0]) for e in batch])
				new_states_img1 = new_states_img[:,0,:,:,np.newaxis]
				new_states_img2 = new_states_img[:,1,:,:,np.newaxis]
				new_states_img3 = new_states_img[:,2,:,:,np.newaxis]
				new_states_img4 = new_states_img[:,3,:,:,np.newaxis]
				
				new_states_oth = np.reshape(np.asarray([np.array(e[3][1:]) for e in batch]), (BATCH_SIZE, n_frames*n_other_var,))

				dones = np.asarray([e[4] for e in batch])
				y_t = np.asarray([e[2] for e in batch])
				
				target_q_values = critic.target_model.predict([new_states_img1, new_states_img2, new_states_img3, new_states_img4, new_states_oth, actor.target_model.predict([new_states_img1, new_states_img2, new_states_img3, new_states_img4, new_states_oth])])  
				q_values = critic.target_model.predict([new_states_img1, new_states_img2, new_states_img3, new_states_img4, new_states_oth, actions])

				#Do the batch update
				
				#print(l, y_t.shape, j)

				for k in range(BATCH_SIZE):

					if dones[k]:
						y_t[k] = rewards[k]

					else:
						y_t[k] = rewards[k] + GAMMA*target_q_values[k]

				Exp.priority_update(indices, abs(target_q_values.reshape(-1)-y_t))

				if (train_indicator):
					
					loss += critic.model.train_on_batch([states_img1, states_img2, states_img3, states_img4, states_oth, actions], y_t, sample_weight=weights) 

					a_for_grad = actor.model.predict([states_img1, states_img2, states_img3, states_img4, states_oth])
					grads = critic.gradients(states_img1, states_img2, states_img3, states_img4, states_oth, a_for_grad)

					actor.train(states_img1, states_img2, states_img3, states_img4, states_oth, grads)
					actor.target_train()
					critic.target_train()
				
				total_reward += R_t
				S_t = S_t1
				R_t = 0
				s_t1 = []
				
			#print("Episode", i, "Step", step, "Action", a_t, "Reward", r_t, "Loss", loss)
			#env.save_screenshots()

			step += 1
			if done:
				break
			 
		if np.mod(i, 5) == 0:
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

#def add_to_state_stack



if __name__ == "__main__":
 
	Simulate_Game()
	