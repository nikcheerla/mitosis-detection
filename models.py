
#various Image<->Image Deep CNN models

import numpy as np

from keras.layers import Input, Reshape, Permute, Flatten, Dense, Lambda, Dropout, merge
from keras.layers import Convolution2D, MaxPooling2D, UpSampling2D, ZeroPadding2D, SpatialDropout2D, Cropping2D
from keras.models import Model, load_model
from keras.optimizers import SGD, adadelta
from keras.callbacks import ProgbarLogger, RemoteMonitor, ReduceLROnPlateau, ModelCheckpoint
from keras import backend as K
from keras import objectives

from dnn import ResNet50

import IPython






class FCNModel(object):
	def __init__(self, local_model, checkpoint="results/checkpoint.h5"):
		self.local_model = local_model
		self.checkpoint = "weights.hdf5"

	def create_subnet(self, input_size=None, output_size=None):
		if output_size is not None:
			L = output_size
			H = output_size * 1000
			while (L < H):
				mid = (L + H)//2
				output_size_pred = self.create_subnet(input_size=mid).output_shape[-1]
				if output_size_pred >= output_size:
					H = mid
				else:
					L = mid + 1

			model = self.create_subnet(input_size=L)
			print (model.output_shape)
			return model

		input_img = Input(shape=(3, input_size, input_size))
		x = self.local_model(input_img)
		model = Model(input_img, x)
		x = Reshape(model.output_shape[1:]) (x)
		model = Model(input_img, x)

		return model

	def train(X, Y, input_size, epochs=5):
		model = create_subnet(input_shape=input_shape)
		model.fit(X, Y, nb_epoch=epoch)

	def evaluate(X, output_size):
		model = create_subnet(output_size=output_size)
		pad_width = (X.shape[-1] - output_size)//2
		X = np.pad(X, pad_width)
		print (X.shape, model.input_shape)

		preds = model.predict(X)
		return preds






if __name__ == "__main__":
	dnn = ResNet50(include_top=False, weights='imagenet')

	model = FCNModel(local_model=dnn)
	m = model.create_subnet(output_size=1000)
	IPython.embed()











"""


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



"""



"""

# Basic Image--> Image model

class ImgToImgModel(object):

	def __init__(self, window_size):
		self.window_size = window_size
		self.metrics = ['binary_accuracy', 'precision', 'recall']
		self.checkpoint = "weights.hdf5"
		self.model = self.build_model()
	
	def build_model(self):
		return None

	def train(self, image_generator, epochs=[5, 5, 5, 5]):
		remote = RemoteMonitor(root='https://localhost:9000')
		reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, verbose=1,
                  patience=3, min_lr=0.001)
		checkpointer = ModelCheckpoint(filepath=self.checkpoint, verbose=1, save_best_only=False)

		image_generator.model = self
		
		for i, nb_epoch in enumerate(epochs):
			print ("ERA {}: ".format(i))

			X_train, y_train = image_generator.data(mode='train')
			print (X_train.shape, y_train.shape)
			self.model.fit(X_train, y_train,
	                nb_epoch=nb_epoch, batch_size=image_generator.batch_size,
	                verbose=1, validation_data=image_generator.data(mode='val'),
	                callbacks=[reduce_lr]
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
				snapshots.append(image[(x - blackout):(x + window_size + blackout), (y - blackout):(y + window_size + blackout)])
		
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
		input_img = Input(shape=(1, self.window_size, self.window_size))

		x = Convolution2D(32, 5, 5, activation='relu', border_mode='same')(input_img)
		x = Convolution2D(16, 5, 5, activation='relu', border_mode='same')(x)
		x = Convolution2D(8, 5, 5, activation='relu', border_mode='same')(x)
		x = Convolution2D(1, 5, 5, activation='relu', border_mode='same')(x)
		
		decoded = constrain(x, input_img)

		autoencoder = Model(input_img, decoded)
		sgd = SGD(lr=1e-2, decay=1e-6, momentum=0.9, nesterov=True)
		autoencoder.compile(optimizer=sgd, loss='mse', metrics=self.metrics)
		autoencoder.summary()

		return autoencoder







#SimpleAutoEncoder image model

class SimpleAutoEncoder(ImgToImgModel):

	def __init__(self, *args, **kwargs):
		super(SimpleAutoEncoder, self).__init__(*args, **kwargs)
	
	def build_model(self):
		input_img = Input(shape=(3, self.window_size, self.window_size))

		x = Convolution2D(16, 8, 8, activation='relu', border_mode='same')(input_img)
		x = MaxPooling2D((2, 2), border_mode='same')(x)
		x = Convolution2D(8, 5, 5, activation='relu', border_mode='same')(x)
		x = MaxPooling2D((2, 2), border_mode='same')(x)
		x = Convolution2D(8, 3, 3, activation='relu', border_mode='same')(x)
		encoded = MaxPooling2D((2, 2), border_mode='same')(x)

		# at this point the representation is (8, 4, 4) i.e. 128-dimensional

		x = Convolution2D(8, 3, 3, activation='relu', border_mode='same')(encoded)
		x = UpSampling2D((2, 2))(x)
		x = Convolution2D(8, 3, 3, activation='relu', border_mode='same')(x)
		x = UpSampling2D((2, 2))(x)
		x = Convolution2D(16, 5, 5, activation='relu', border_mode='same')(x)
		x = UpSampling2D((2, 2))(x)
		#x = ZeroPadding2D((8, 8))(x)
		decoded = constrain(Convolution2D(1, 8, 8, activation='sigmoid', border_mode='same')(x), input_img)

		autoencoder = Model(input_img, decoded)
		autoencoder.compile(optimizer=adadelta(lr=0.4), loss='binary_crossentropy', metrics=self.metrics)
		autoencoder.summary()

		return autoencoder











#VGGNetEncoder model

class VGGNetEncoder(ImgToImgModel):

	def __init__(self, *args, **kwargs):
		self.map_size = kwargs.pop('map_size', 64)
		super(VGGNetEncoder, self).__init__(*args, **kwargs)
	
	def build_model(self):
		input_img = Input(shape=(1, self.window_size, self.window_size))
		map_size = self.map_size
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
		x = link3 = MaxPooling2D((2,2), strides=(2,2))(x)

		x = Convolution2D(map_size/2, 3, 3, activation='relu', border_mode='same')(x)
		x = Convolution2D(map_size/2, 3, 3, activation='relu', border_mode='same')(x)
		x = Convolution2D(map_size/2, 3, 3, activation='softplus', border_mode='same')(x)
		x = Dropout(p=0.3)(x)
		x = link4 = MaxPooling2D((2,2), strides=(2,2))(x)

		x = Convolution2D(map_size, 3, 3, activation='relu', border_mode='same')(x)
		x = Convolution2D(map_size, 3, 3, activation='relu', border_mode='same')(x)
		x = Convolution2D(map_size, 3, 3, activation='softplus', border_mode='same')(x)
		x = link5 = MaxPooling2D((2,2), strides=(2,2))(x)

		x = Flatten()(x)
		x = Dense(4*self.window_size/32*self.window_size/32, activation='sigmoid')(x)
		x = Dropout(0.4)(x)
		x = Reshape((4, self.window_size/32, self.window_size/32))(x)

		x = project(x, link5, input_l=input_img)
		x = Convolution2D(map_size, 3, 3, activation='softplus', border_mode='same', init='glorot_uniform')(x)
		x = Convolution2D(map_size, 3, 3, activation='relu', border_mode='same', init='glorot_uniform')(x)
		x = Convolution2D(map_size, 3, 3, activation='relu', border_mode='same')(x)
		x = UpSampling2D((2, 2))(x)

		#x = project(x, link4, input_l=input_img)
		x = Convolution2D(map_size/2, 3, 3, activation='softplus', border_mode='same')(x)
		x = Convolution2D(map_size/2, 3, 3, activation='relu', border_mode='same')(x)
		x = Convolution2D(map_size/2, 3, 3, activation='relu', border_mode='same')(x)
		x = UpSampling2D((2, 2))(x)

		#x = project(x, link3, input_l=input_img)
		x = Convolution2D(map_size/2, 3, 3, activation='softplus', border_mode='same')(x)
		x = Convolution2D(map_size/2, 3, 3, activation='relu', border_mode='same')(x)
		x = Convolution2D(map_size/2, 3, 3, activation='relu', border_mode='same')(x)
		x = UpSampling2D((2, 2))(x)

		#x = project(x, link2, input_l=input_img)
		x = Convolution2D(map_size/4, 3, 3, activation='softplus', border_mode='same')(x)
		x = Convolution2D(map_size/4, 3, 3, activation='relu', border_mode='same')(x)
		x = UpSampling2D((2, 2))(x)

		#x = project(x, link1, input_l=input_img)
		x = Convolution2D(map_size/4, 3, 3, activation='softplus', border_mode='same')(x)
		x = Convolution2D(map_size/4, 3, 3, activation='relu', border_mode='same')(x)
		x = UpSampling2D((2, 2))(x)
		#x = project(x, input_img, input_l=input_img)
		x = Convolution2D(1, 8, 8, activation='relu', border_mode='same')(x)
		decoded = constrain(x, input_img)

		autoencoder = Model(input_img, decoded)
		sgd = SGD(lr=1e-2, decay=1e-6, momentum=0.9, nesterov=True)
		autoencoder.compile(optimizer=sgd, loss='binary_crossentropy', metrics=self.metrics)
		autoencoder.summary()

		return autoencoder
	










VAE model (implements AbstractModel from models.py) that autoencodes single state

class VAEModel(ImgToImgModel):

	def __init__(self, *args, **kwargs):
		self.batch_size = kwargs.pop('batch_size', 100)
		self.latent_dim = kwargs.pop('latent_dim', 12)
		self.std = kwargs.pop('std', 0.06)
		self.note_dim = kwargs.pop('note_dim', 1)
		super(VAEModel, self).__init__(*args, **kwargs)



	def build_model(self):

		input_img = Input(batch_shape=(self.batch_size, 1, self.window_size, self.window_size))

		encoder = self.build_encoder()
		decoder = self.build_decoder()

		z_mean, z_log_var = encoder(input_img)

		def sampling(args):
			z_mean, z_log_var = args
			epsilon = K.random_normal(shape=(self.batch_size, self.latent_dim), mean=0.,
				std=self.std)
			return z_mean + K.exp(z_log_var / 2) * epsilon

		z = Lambda(sampling, output_shape=(self.latent_dim,)) ([z_mean, z_log_var])
		output = decoder(z)
		output = Reshape ((1, self.window_size, self.window_size)) (output)

		model = Model(input_img, output)
		print ("\nCombined model:")
		model.summary()

		def vae_loss(x, x_decoded_mean):
			xent_loss = self.window_size * self.window_size * objectives.binary_crossentropy(x, x_decoded_mean)
			kl_loss = - 0.5 * K.sum(1 + z_log_var[0] - K.square(z_mean[0]) - K.exp(z_log_var[0]), axis=-1)
			return xent_loss

		sgd = SGD(lr=7e-3, decay=1e-6, momentum=0.8, nesterov=True)
		model.compile(optimizer=sgd, loss=vae_loss, metrics=['binary_crossentropy', 'mse'])

		return model



	def build_encoder(self):
		input_img = Input(shape=(1, self.window_size, self.window_size))
		x = Convolution2D(50, 3, 3, activation='relu', border_mode='same') (input_img)
		x = Convolution2D(50, 3, 3, activation='relu', border_mode='same') (x)
		x = MaxPooling2D((2, 2)) (x)
		x = Convolution2D(50, 3, 3, activation='softplus', border_mode='same') (x)
		x = Convolution2D(50, 3, 3, activation='softplus', border_mode='same') (x)
		x = MaxPooling2D((2, 2)) (x)
		x = Convolution2D(50, 5, 5, activation='relu', border_mode='same') (x)
		x = Convolution2D(50, 5, 5, activation='tanh', border_mode='same') (x)
		x = MaxPooling2D((2, 2)) (x)
		x = Convolution2D(1, 1, 1, activation='tanh', border_mode='same') (x)
		x = Flatten() (x)

		#h = Dense(intermediate_dim, activation='relu') (x)

		z_mean = Dense(self.latent_dim)(x)
		z_log_var = Dense(self.latent_dim)(x)

		self.encoder_input = input_img
		self.encoder = Model(input_img, [z_mean, z_log_var])

		print ("\nEncoder Model:")
		self.encoder.summary()
		return self.encoder



	def build_decoder(self):
		input_img = Input(shape=(self.latent_dim,))

		x = Dense(self.window_size*self.window_size/64, activation='tanh') (input_img)
		x = Reshape ((1, self.window_size/8, self.window_size/8)) (x)

		x = UpSampling2D((2, 2)) (x)
		x = Convolution2D(10, 3, 3, activation='tanh', border_mode='same') (x)
		x = Convolution2D(20, 3, 3, activation='relu', border_mode='same') (x)
		
		x = UpSampling2D((2, 2)) (x)
		x = Convolution2D(30, 3, 3, activation='softplus', border_mode='same') (x)
		x = Convolution2D(40, 3, 3, activation='softplus', border_mode='same') (x)
		
		x = UpSampling2D((2, 2)) (x)
		x = Convolution2D(30, 5, 5, activation='relu', border_mode='same') (x)
		x = Convolution2D(1, 5, 5, activation='relu', border_mode='same') (x)
		
		self.decoder = Model(input_img, x)
		print ("\nDecoder Model:")
		self.decoder.summary()
		return self.decoder






"""





