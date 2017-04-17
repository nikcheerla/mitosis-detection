import numpy as np
import matplotlib.pyplot as plt

import keras
from keras.utils import np_utils
from keras.models import Model
from keras.layers import Conv2D, MaxPooling2D, Input, Flatten, Dense, AveragePooling2D

import IPython

from external.vgg19 import VGG19

from dnn import dilation_map



"""
input_img = Input(shape=(80, 80, 3))
x = Conv2D(50, (3, 3), padding='same', activation='relu')(input_img)
x = MaxPooling2D((3, 3)) (x)
x = Conv2D(30, (3, 3), padding='same', activation='relu')(x)
x = MaxPooling2D((3, 3)) (x)
x = Conv2D(15, (3, 3), padding='same', activation='relu')(x)
x = MaxPooling2D((3, 3)) (x)
x = Conv2D(8, (3, 3), padding='same', activation='relu')(x)
x = AveragePooling2D((2, 2)) (x)
x = Conv2D(2, (1, 1), padding='same', activation='softmax')(x)

model = Model(input_img, x)
model.summary()
model.compile(optimizer='adadelta', loss='binary_crossentropy',  metrics=['accuracy'])
"""




model = VGG19(weights=None, input_shape=(80, 80, 3), classes=2, filter_size=16, pooling='avg')
model.summary()
sgd = keras.optimizers.SGD(lr=1e-2, decay=1e-6, momentum=0.8, nesterov=True)

model.compile(optimizer=sgd, loss='binary_crossentropy',  metrics=['accuracy'])



data = np.load("results/train_data_icpr.npy.npz")
X, Y = data["X"], data["Y"]


Y = np.array([Y, 1 - Y]).swapaxes(0, 4)[0]
print (Y.shape)

model.fit(X, Y, epochs=3, batch_size=32)


model2 = dilation_map(model)
model2.summary()

IPython.embed()