import numpy as np
from keras import Sequential
import keras.backend as K
import tensorflow as tf
from keras.layers import Input, Conv2D, MaxPooling2D, Flatten,Lambda, concatenate
from keras.initializers import VarianceScaling
import cv2



class ImageDataProcessing(object):


	def __init__(self, sess, n_other_var, n_frames):

		self.sess = sess
		K.set_session(sess)

		self.to_resize = 82
		self.img_rows = self.to_resize
		self.img_cols = self.to_resize
		self.model = Sequential()
		self.sess.run(tf.initialize_all_variables())
		self.img_channels = 1
		self.batch = 1
		self.n_oth_var = n_other_var
		self.no_input_images = n_frames
		#s_t1 = np.append(x_t1, s_t[:, :3, :, :], axis=1)

	def convert_image_to_1d(self, image):

		return np.array(image).reshape(-1)

	def convert_1d_to_image(self, image_data):

		return image_data.reshape(self.to_resize, self.to_resize)

	def state_to_img_n_other_var_nn(self, s_t):

		image1_t = s_t[0][0].reshape((-1, self.img_rows, self.img_cols, 1))
		image2_t = s_t[0][1].reshape((-1, self.img_rows, self.img_cols, 1))
		image3_t = s_t[0][2].reshape((-1, self.img_rows, self.img_cols, 1))
		image4_t = s_t[0][3].reshape((-1, self.img_rows, self.img_cols, 1))
		
		other_state_var_t = np.asarray(s_t[1:]).reshape((-1, self.no_input_images* self.n_oth_var))

		return image1_t, image2_t, image3_t, image4_t, other_state_var_t


	def build_CNN_model(self):

		print("Now we build the CNN model")
		
		I1 = Input(shape=(self.img_rows,self.img_cols,self.img_channels))
		I2 = Input(shape=(self.img_rows,self.img_cols,self.img_channels))
		I3 = Input(shape=(self.img_rows,self.img_cols,self.img_channels))
		I4 = Input(shape=(self.img_rows,self.img_cols,self.img_channels))
		
		I = concatenate([I1,I2, I3, I4],axis=-1)	

		M1 = Conv2D(16, kernel_size=(4, 4), strides=(2,2), activation='relu',kernel_initializer=lambda shape:VarianceScaling(scale=1e-2)(shape), padding='same')(I)
		M2 = Conv2D(16, kernel_size=(4, 4), strides=(2,2), activation='relu', kernel_initializer=lambda shape:VarianceScaling(scale=1e-2)(shape), padding='same')(M1)
		M3 = Conv2D(16, kernel_size=(4, 4), strides=(2,2), activation='relu', kernel_initializer=lambda shape:VarianceScaling(scale=1e-2)(shape), padding='same')(M2)
		M4 = Conv2D(16, kernel_size=(4, 4), strides=(2,2), activation='relu', kernel_initializer=lambda shape:VarianceScaling(scale=1e-2)(shape), padding='same')(M3)
		
		S = Flatten()(M4)

		return S, I1, I2, I3, I4
'''
	def add_to_state(self, s_t, S_t):

		



	def draw_lane_lines(self, image):

		imshape = image.shape
		
		# Greyscale image
		greyscaled_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

		# Gaussian Blur
		blurred_grey_image = cv2.GaussianBlur(greyscaled_image, (5,5), cv2.BORDER_DEFAULT)
		
		# Canny edge detection
		edges_image = cv2.Canny(blurred_grey_image, 50, 150)
		
		# Mask edges image
		border = 0
		vertices = np.array([[(0,imshape[0]),(imshape[1]/2, imshape[0]/2), (imshape[0], imshape[1])]], dtype=np.int32)
		edges_image_with_mask = cv2.selectROI(vertices,edges_image)
		
		# Hough lines
		rho = 2 
		theta = np.pi/180 
		threshold = 45    
		min_line_len = 40
		max_line_gap = 100 

		lines_image = cv2.HoughLines(edges_image_with_mask, rho, theta, threshold, min_line_len, max_line_gap)

		# Convert Hough from single channel to RGB to prep for weighted
		hough_rgb_image = cv2.cvtColor(lines_image, cv2.COLOR_GRAY2BGR)
	 
		# Combine lines image with original image
		final_image = weighted_img(hough_rgb_image, image)
		
		return final_image
'''