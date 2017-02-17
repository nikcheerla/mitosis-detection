
#various Image<->Image Deep CNN models

import numpy as np

from keras.layers import Input, Dense, Dropout, Reshape, Flatten, merge
from keras.layers import Convolution2D, MaxPooling2D, UpSampling2D, ZeroPadding2D, SpatialDropout2D, Cropping2D
from keras.models import Model, load_model
from keras.optimizers import SGD, adadelta
from keras.callbacks import ProgbarLogger, RemoteMonitor, ReduceLROnPlateau, ModelCheckpoint

from keras import backend as K





# Utility keras connector methods

def project(l1, l2, input_l = None):
	w1 = Model(input_l, l1).layers[-1].output_shape[2]
	w2 = Model(input_l, l2).layers[-1].output_shape[2]
	#print (w1, w2)

	if w2 < w1:
		s2 = (w1 - w2)//2
		l2 = ZeroPadding2D((s2, w1 - w2 - s2, s2, w1 - w2 - s2))(l2)
	elif w1 < w2:
		s1 = (w2 - w1)//2
		#print (s1)
		l1 = ZeroPadding2D((s1, w2 - w1 - s1, s1, w2 - w1 - s1))(l1)
	
	return merge([l1, l2], mode='concat', concat_axis=1)

def constrain(l1, input_l = None):
	model = Model(input_l, l1)

	w1 = model.layers[-1].output_shape[2]
	i1 = model.layers[0].batch_input_shape[2]

	if w1 == i1:
		return l1

	s1 = (i1 - w1)//2
	#print ("Inputs", w1, i1)
	w1 = Cropping2D(cropping=((-s1, s1 + w1 - i1), (-s1, s1 + w1 - i1)))(l1)
	return w1

def binary_crossentropy_norm(y_true, y_pred):
    return K.mean(K.binary_crossentropy(y_pred, y_true, from_logits=True), axis=-1)	









# Basic Image--> Image model

class ImgToImgModel(object):

	def __init__(self, window_size):
		self.window_size = window_size
		self.metrics = ['binary_accuracy', 'precision', 'recall']
		self.checkpoint = "weights.hdf5"
		self.model = self.build_model()
	
	def build_model(self):
		return None

	def train(self, image_generator, epochs=20):
		remote = RemoteMonitor(root='https://localhost:9000')
		reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, verbose=1,
                  patience=3, min_lr=0.001)
		checkpointer = ModelCheckpoint(filepath=self.checkpoint, verbose=1, save_best_only=False)

		image_generator.model = self
		
		for i in range(1, epochs + 1):
			print ("Epoch {}: ".format(i))

			X_train, y_train = image_generator.data(mode='train')
			self.model.fit(X_train, y_train,
	                nb_epoch=1, batch_size=image_generator.batch_size,
	                verbose=1, validation_data=image_generator.data(mode='val'),
	                callbacks=[reduce_lr, remote, checkpointer]
	            )
			image_generator.checkpoint()


	def predict(self, image_array, batch_size=32, verbose=False):
		return self.model.predict(image_array, batch_size=batch_size, verbose=verbose)

	def evaluate(self, image, overlap=0, blackout=20):

		window_size = self.window_size - 2*blackout

		width, height = image.shape[0], image.shape[1]
		x_ind = np.arange(window_size, width - 2*window_size, window_size - overlap).astype(int)
		y_ind = np.arange(window_size, height - 2*window_size, window_size - overlap).astype(int)

		hmap = np.zeros((width, height))
		coverage = np.zeros((width, height))

		snapshots = []
		for x in x_ind:
			for y in y_ind:
				print (x, y)
				snapshots.append(image[(x - blackout):(x + window_size + blackout), (y - blackout):(y + window_size + blackout), :])
		
		snapshots = np.array(snapshots)
		print (snapshots.shape)
		snapshots = np.rollaxis(snapshots, 3, 1)
		preds = self.predict(snapshots, batch_size=32, verbose=1)

		i = 0
		for x in x_ind:
			for y in y_ind:
				print (x, y)
				hmap[x:(x + window_size), y:(y + window_size)] += preds[i, 0][blackout:(self.window_size - blackout), blackout:(self.window_size - blackout)]
				coverage[x:(x + window_size), y:(y + window_size)] += 1
				i+=1
				#TODO: implement averaging at boundaries

		return hmap/coverage

	def save(self, weights_file):
		self.model.save(weights_file)

	@classmethod
	def load(self, weights_file):

		model = load_model(weights_file, custom_objects={'binary_crossentropy_norm': binary_crossentropy_norm})

		window_size = model.layers[0].batch_input_shape[2]
		dnn = self(window_size)
		dnn.model = model

		return dnn







#Simple conv filter image model

class SimpleConvFilter(ImgToImgModel):

	def __init__(self, *args, **kwargs):
		super(SimpleConvFilter, self).__init__(*args, **kwargs)
	
	def build_model(self):
		input_img = Input(shape=(3, self.window_size, self.window_size))

		x = Convolution2D(32, 5, 5, activation='relu', border_mode='same')(input_img)
		x = Convolution2D(16, 5, 5, activation='relu', border_mode='same')(x)
		x = Convolution2D(8, 5, 5, activation='relu', border_mode='same')(x)
		x = Convolution2D(1, 5, 5, activation='relu', border_mode='same')(x)
		
		decoded = constrain(x, input_img)

		autoencoder = Model(input_img, decoded)
		sgd = SGD(lr=1e-2, decay=1e-6, momentum=0.9, nesterov=True)
		autoencoder.compile(optimizer=sgd, loss='binary_crossentropy', metrics=self.metrics)
		autoencoder.summary()

		return autoencoder







#SimpleAutoEncoder image model

class SimpleAutoEncoder(ImgToImgModel):

	def __init__(self, *args, **kwargs):
		super(SimpleAutoEncoder, self).__init__(*args, **kwargs)
	
	def build_model(self):
		input_img = Input(shape=(3, self.window_size, self.window_size))

		x = Convolution2D(16, 8, 8, activation='relu', border_mode='same')(input_img)
		x = MaxPooling2D((4, 4), border_mode='same')(x)
		link2 = Convolution2D(8, 5, 5, activation='relu', border_mode='same')(x)
		x = MaxPooling2D((3, 3), border_mode='same')(link2)
		x = Convolution2D(8, 3, 3, activation='relu', border_mode='same')(x)
		encoded = MaxPooling2D((2, 2), border_mode='same')(x)

		# at this point the representation is (8, 4, 4) i.e. 128-dimensional

		x = Convolution2D(8, 3, 3, activation='relu', border_mode='same')(encoded)
		x = UpSampling2D((2, 2))(x)
		x = Convolution2D(8, 3, 3, activation='relu', border_mode='same')(x)
		x = UpSampling2D((3, 3))(x)
		x = project(x, link2, input_l=input_img)
		x = Convolution2D(16, 5, 5, activation='relu', border_mode='same')(x)
		x = UpSampling2D((4, 4))(x)
		#x = ZeroPadding2D((8, 8))(x)
		decoded = constrain(Convolution2D(1, 8, 8, activation='sigmoid', border_mode='same')(x), input_img)

		autoencoder = Model(input_img, decoded)
		autoencoder.compile(optimizer=adadelta(lr=0.4), loss='binary_crossentropy', metrics=self.metrics)
		autoencoder.summary()

		return autoencoder











#VGGNetEncoder model

class VGGNetEncoder(ImgToImgModel):

	def __init__(self, *args, **kwargs):
		super(VGGNetEncoder, self).__init__(*args, **kwargs)
	
	def build_model(self):
		input_img = Input(shape=(3, self.window_size, self.window_size))
		map_size = 96
		x = Convolution2D(map_size/8, 3, 3, activation='relu', border_mode='same')(input_img)
		x = Convolution2D(map_size/4, 3, 3, activation='softplus', border_mode='same')(x)
		x = SpatialDropout2D(p=0.4)(x)
		x = link1 = MaxPooling2D((2,2), strides=(2,2))(x)

		x = Convolution2D(map_size/4, 3, 3, activation='relu', border_mode='same')(x)
		x = Convolution2D(map_size/4, 3, 3, activation='softplus', border_mode='same')(x)
		x = SpatialDropout2D(p=0.4)(x)
		x = link2 = MaxPooling2D((2,2), strides=(2,2))(x)

		x = Convolution2D(map_size/2, 3, 3, activation='relu', border_mode='same')(x)
		x = Convolution2D(map_size/2, 3, 3, activation='relu', border_mode='same')(x)
		x = Convolution2D(map_size/2, 3, 3, activation='softplus', border_mode='same')(x)
		x = Dropout(p=0.3)(x)
		x = link3 = MaxPooling2D((2,2), strides=(2,2))(x)

		x = Convolution2D(map_size/2, 3, 3, activation='relu', border_mode='same')(x)
		x = Convolution2D(map_size/2, 3, 3, activation='relu', border_mode='same')(x)
		x = Convolution2D(map_size/2, 3, 3, activation='softplus', border_mode='same')(x)
		x = Dropout(p=0.3)(x)
		x = link4 = MaxPooling2D((2,2), strides=(2,2))(x)

		x = Convolution2D(map_size, 3, 3, activation='relu', border_mode='same')(x)
		x = Convolution2D(map_size, 3, 3, activation='relu', border_mode='same')(x)
		x = Convolution2D(map_size, 3, 3, activation='softplus', border_mode='same')(x)
		x = Dropout(p=0.2)(x)
		x = link5 = MaxPooling2D((2,2), strides=(2,2))(x)

		x = Flatten()(x)
		x = Dense(256, activation='sigmoid')(x)
		x = Dropout(0.2)(x)
		x = Reshape((4, 8, 8))(x)

		x = project(x, link5, input_l=input_img)
		x = Convolution2D(map_size, 3, 3, activation='softplus', border_mode='same', init='glorot_uniform')(x)
		x = Convolution2D(map_size, 3, 3, activation='relu', border_mode='same', init='glorot_uniform')(x)
		x = Convolution2D(map_size, 3, 3, activation='relu', border_mode='same')(x)
		x = UpSampling2D((2, 2))(x)

		x = project(x, link4, input_l=input_img)
		x = Convolution2D(map_size/2, 3, 3, activation='softplus', border_mode='same')(x)
		x = Convolution2D(map_size/2, 3, 3, activation='relu', border_mode='same')(x)
		x = Convolution2D(map_size/2, 3, 3, activation='relu', border_mode='same')(x)
		x = UpSampling2D((2, 2))(x)

		x = project(x, link3, input_l=input_img)
		x = Convolution2D(map_size/2, 3, 3, activation='softplus', border_mode='same')(x)
		x = Convolution2D(map_size/2, 3, 3, activation='relu', border_mode='same')(x)
		x = Convolution2D(map_size/2, 3, 3, activation='relu', border_mode='same')(x)
		x = UpSampling2D((2, 2))(x)

		x = project(x, link2, input_l=input_img)
		x = Convolution2D(map_size/4, 3, 3, activation='softplus', border_mode='same')(x)
		x = Convolution2D(map_size/4, 3, 3, activation='relu', border_mode='same')(x)
		x = UpSampling2D((2, 2))(x)

		x = project(x, link1, input_l=input_img)
		x = Convolution2D(map_size/4, 3, 3, activation='softplus', border_mode='same')(x)
		x = Convolution2D(map_size/4, 3, 3, activation='relu', border_mode='same')(x)
		x = UpSampling2D((2, 2))(x)
		x = project(x, input_img, input_l=input_img)
		x = Convolution2D(1, 8, 8, activation='relu', border_mode='same')(x)
		decoded = constrain(x, input_img)

		autoencoder = Model(input_img, decoded)
		sgd = SGD(lr=1e-2, decay=1e-6, momentum=0.9, nesterov=True)
		autoencoder.compile(optimizer=sgd, loss=binary_crossentropy_norm, metrics=self.metrics)
		autoencoder.summary()

		return autoencoder
	






if __name__ == "__main__":
	model = VGGNetEncoder(256)






