from autoencoder import Autoencoder

autoencoder = Autoencoder(filters=[60,30], kernel_size=(10,10), poll_size=(6,6), input_size=(360,360,1), epochs = 5, batch_size= 16)
autoencoder.createNetwork()

autoencoder.load_weights()

autoencoder.predict('test.png') 
