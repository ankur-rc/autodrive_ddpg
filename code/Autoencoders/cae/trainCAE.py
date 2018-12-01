from autoencoder import Autoencoder

autoencoder = Autoencoder(filters=[60,30], kernel_size=(10,10), poll_size=(6,6), input_size=(360,360,1), epochs = 5, batch_size= 16)
autoencoder.createNetwork()

autoencoder.getData('/home/santosh/Carla_0.8.2/Setting-Up-CARLA-RL/rl_project/screens/rgb/')

autoencoder.train()
