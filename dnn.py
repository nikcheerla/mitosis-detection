# -*- coding: utf-8 -*-
'''ResNet50 model for Keras.

# Reference:

- [Deep Residual Learning for Image Recognition](https://arxiv.org/abs/1512.03385)

Adapted from code contributed by BigMoyan.
'''

import numpy as np
import warnings, json

import keras
from keras.layers import merge, Input
from keras.layers import Dense, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D, ZeroPadding2D, AveragePooling2D, UpSampling2D, Cropping2D

from keras.layers import BatchNormalization
from keras.models import Model
from keras.preprocessing import image
import keras.backend as K
from keras.utils.layer_utils import convert_all_kernels_in_model
from keras.utils.data_utils import get_file
from keras.applications.imagenet_utils import preprocess_input

from external.resnet50 import ResNet50

import IPython







""" Weights utils """

import h5py

def str_shape(x):
    return 'x'.join(map(str, x.shape))


def load_weights(model, filepath, lookup={}, ignore=[], transform=None, verbose=True):
    """Modified version of keras load_weights that loads as much as it can.
    Useful for transfer learning.
    read the weights of layers stored in file and copy them to a model layer.
    the name of each layer is used to match the file's layers with the model's.
    It is possible to have layers in the model that dont appear in the file..
    The loading stopps if a problem is encountered and the weights of the
    file layer that first caused the problem are returned.
    # Arguments
        model: Model
            target
        filepath: str
            source hdf5 file
        lookup: dict (optional)
            by default, the weights of each layer in the file are copied to the
            layer with the same name in the model. Using lookup you can replace
            the file name with a different model layer name, or to a list of
            model layer names, in which case the same weights will be copied
            to all layer models.
        ignore: list (optional)
            list of model layer names to ignore in
        transform: None (optional)
            This is an optional function that receives the list of weighs
            read from a layer in the file and the model layer object to which
            these weights should be loaded.
        verbose: bool
            high recommended to keep this true and to follow the print messages.
    # Returns
        weights of the file layer which first caused the load to abort or None
        on successful load.
    """
    if verbose:
        print ('Loading', filepath, 'to', model.name)
    flattened_layers = model.layers
    with h5py.File(filepath, mode='r') as f:
        # new file format
        layer_names = [n.decode('utf8') for n in f.attrs['layer_names']]

        # we batch weight value assignments in a single backend call
        # which provides a speedup in TensorFlow.
        weight_value_tuples = []
        for name in layer_names:
            if verbose:
                print name,
            g = f[name]
            weight_names = [n.decode('utf8') for n in
                            g.attrs['weight_names']]
            if len(weight_names):
                weight_values = [g[weight_name] for weight_name in
                                 weight_names]
                if verbose:
                    print 'loading', ' '.join(
                            str_shape(w) for w in weight_values),
                target_names = lookup.get(name, name)
                if isinstance(target_names, basestring):
                    target_names = [target_names]
                # handle the case were lookup asks to send the same weight to multiple layers
                target_names = [target_name for target_name in target_names if
                                target_name == name or target_name not in layer_names]
                for target_name in target_names:
                    if verbose:
                        print target_name,
                    try:
                        layer = model.get_layer(name=target_name)
                    except:
                        layer = None
                    if layer:
                        # the same weight_values are copied to each of the target layers
                        symbolic_weights = layer.trainable_weights + layer.non_trainable_weights

                        if transform is not None:
                            transformed_weight_values = transform(weight_values, layer)
                            if transformed_weight_values is not None:
                                if verbose:
                                    print '(%d->%d)'%(len(weight_values),len(transformed_weight_values)),
                                weight_values = transformed_weight_values

                        problem = len(symbolic_weights) != len(weight_values)
                        if problem and verbose:
                            print '(bad #wgts)',
                        if not problem:
                            weight_value_tuples += zip(symbolic_weights, weight_values)
                    else:
                        problem = True
                    if problem:
                        if verbose:
                            if name in ignore or ignore == '*':
                                print '(skipping)',
                            else:
                                print 'ABORT'
                        if not (name in ignore or ignore == '*'):
                            K.batch_set_value(weight_value_tuples)
                            return [np.array(w) for w in weight_values]
                if verbose:
                    print
            else:
                if verbose:
                    print 'skipping this is empty file layer'
        K.batch_set_value(weight_value_tuples)























""" DNN FCN utils """

def constrain(l1, inputs):
    model = Model(inputs, l1)

    w1 = model.layers[-1].output_shape[2]
    i1 = model.layers[0].batch_input_shape[2]

    if w1 == i1:
        return l1

    s1 = (i1 - w1)//2
    #print ("Inputs", w1, i1)
    w1 = Cropping2D(cropping=((-s1, s1 + w1 - i1), (-s1, s1 + w1 - i1)))(l1)
    return w1


def fully_convolutional(model, inputs, arr):
    desired_output_shape = model.input_shape

    for i in range(0, len(arr)):
        cur_shape = Model(inputs, arr[i]).output_shape
        scale_factor = desired_output_shape[-1]/cur_shape[-1]
        arr[i] = UpSampling2D((scale_factor, scale_factor)) (arr[i])
        #arr[i] = constrain(arr[i], inputs)

    x = merge(arr, mode='concat', concat_axis=1)
    return Model(inputs, x)













if __name__ == '__main__':
    model, inputs, arr = ResNet50(include_top=False, weights='imagenet', input_shape=(3, 224, 224))
    model2 = fully_convolutional(model, inputs, arr)
    model2.summary()

    
    img_path = 'external/cats.jpg'
    img = image.load_img(img_path, target_size=(224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    print('Input image shape:', x.shape)

    preds = model2.predict(x)
    IPython.embed()
    
