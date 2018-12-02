import numpy as np
from keras import Sequential
import keras.backend as K
import tensorflow as tf
from keras.layers import Input, Conv2D, MaxPooling2D, Flatten,Lambda
from keras.initializers import VarianceScaling
import cv2



class PreProcessing(object):


	def __init__(self, sess, n_other_var):

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
		#s_t1 = np.append(x_t1, s_t[:, :3, :, :], axis=1)

	def image_preprocess(self, image):

		#skimage.color.rgb2gray(image)
		x_t1 = image[160:360, 0:360]
		x_t1 = cv2.resize(x_t1,(self.to_resize, self.to_resize))#skimage.transform.resize(x_t1,(self.to_resize, self.to_resize))
		x_t1 = cv2.normalize(x_t1, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
		x_t1 = cv2.cvtColor(x_t1, cv2.COLOR_BGR2GRAY)
		#x_t1 = self.draw_lane_lines(x_t1)
		#x_t1 = skimage.exposure.rescale_intensity(x_t1, out_range=(0, 255))
		#cv2.imshow("",x_t1)
		#cv2.waitKey(5)
		#processed_img = x_t1.reshape(1, 1, x_t1.shape[0], x_t1.shape[1])
		
		return np.array(x_t1).reshape(-1)

	def convert_image_to_1d(self, image):

		return np.array(image).reshape(-1)

	def convert_1d_to_image(self, image_data):

		return image_data.reshape(self.to_resize, self.to_resize)

	def state_to_img_n_other_var_nn(self, s_t):

		image_t = self.convert_1d_to_image(s_t[0:self.img_rows**2])
		other_state_var_t = s_t[self.img_rows**2:self.img_rows**2 + self.n_oth_var]

		return image_t.reshape((-1, self.img_rows, self.img_cols, 1)), other_state_var_t.reshape((-1, self.n_oth_var))


	def build_CNN_model(self):

		print("Now we build the CNN model")
		I = Input(shape=(self.img_rows,self.img_cols,self.batch))
		I1 = Conv2D(16, kernel_size=(4, 4), strides=(2,2), activation='relu',kernel_initializer=lambda shape:VarianceScaling(scale=1e-2)(shape), padding='same')(I)
		I2 = Conv2D(16, kernel_size=(4, 4), strides=(2,2), activation='relu', kernel_initializer=lambda shape:VarianceScaling(scale=1e-2)(shape), padding='same')(I1)
		I3 = Conv2D(16, kernel_size=(4, 4), strides=(2,2), activation='relu', kernel_initializer=lambda shape:VarianceScaling(scale=1e-2)(shape), padding='same')(I2)
		I4 = Conv2D(16, kernel_size=(4, 4), strides=(2,2), activation='relu', kernel_initializer=lambda shape:VarianceScaling(scale=1e-2)(shape), padding='same')(I3)
		
		S =Flatten()(I4)

		return S, I#model
'''
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