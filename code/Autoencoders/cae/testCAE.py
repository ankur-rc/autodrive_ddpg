from autoencoder import Autoencoder

autoencoder = Autoencoder(filters=[5,15,30], kernel_size=[(3,3),(5,5),(10,10)], poll_size=(3,3), input_size=(360,360,1), epochs = 20, batch_size= 16)
autoencoder.createNetwork()

autoencoder.load_weights()

autoencoder.predict('test.png') 
