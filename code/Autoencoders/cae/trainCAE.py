from autoencoder import Autoencoder

autoencoder = Autoencoder(filters=[5,15,30], kernel_size=[(3,3),(5,5),(10,10)], poll_size=(3,3), input_size=(360,360,1), epochs = 20, batch_size= 16)
autoencoder.createNetwork()

autoencoder.getData('/home/santosh/Carla_0.8.2/Setting-Up-CARLA-RL/rl_project/screens/rgb/')

autoencoder.train()
