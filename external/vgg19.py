# -*- coding: utf-8 -*-
"""VGG19 model for Keras.
# Reference
- [Very Deep Convolutional Networks for Large-Scale Image Recognition](https://arxiv.org/abs/1409.1556)
"""
from __future__ import print_function
from __future__ import absolute_import

import warnings

import numpy as np

from keras.layers import Input
from keras import layers
from keras.layers import Dense, Activation, Flatten, BatchNormalization
from keras.layers import Conv2D, MaxPooling2D, ZeroPadding2D, AveragePooling2D, GlobalAveragePooling2D, GlobalMaxPooling2D
from keras.layers import Dropout, SpatialDropout2D
from keras.models import Model
from keras import backend as K

from keras.engine.topology import get_source_inputs
from keras.utils import layer_utils
from keras.utils.data_utils import get_file
from keras.applications.imagenet_utils import decode_predictions
from keras.applications.imagenet_utils import preprocess_input
from keras.applications.imagenet_utils import _obtain_input_shape
from keras.preprocessing import image

import IPython


WEIGHTS_PATH = 'https://github.com/fchollet/deep-learning-models/releases/download/v0.1/vgg19_weights_tf_dim_ordering_tf_kernels.h5'
WEIGHTS_PATH_NO_TOP = 'https://github.com/fchollet/deep-learning-models/releases/download/v0.1/vgg19_weights_tf_dim_ordering_tf_kernels_notop.h5'


def VGG19(include_top=None, weights='imagenet',
          input_tensor=None, input_shape=None,
          pooling=None, filter_size=64, dropout=0.2,
          classes=1000):
    """Instantiates the VGG19 architecture.
    Optionally loads weights pre-trained
    on ImageNet. Note that when using TensorFlow,
    for best performance you should set
    `image_data_format="channels_last"` in your Keras config
    at ~/.keras/keras.json.
    The model and the weights are compatible with both
    TensorFlow and Theano. The data format
    convention used by the model is the one
    specified in your Keras config file.
    # Arguments
        include_top: whether to include the 3 fully-connected
            layers at the top of the network.
        weights: one of `None` (random initialization)
            or "imagenet" (pre-training on ImageNet).
        input_tensor: optional Keras tensor (i.e. output of `layers.Input()`)
            to use as image input for the model.
        input_shape: optional shape tuple, only to be specified
            if `include_top` is False (otherwise the input shape
            has to be `(224, 224, 3)` (with `channels_last` data format)
            or `(3, 224, 244)` (with `channels_first` data format).
            It should have exactly 3 inputs channels,
            and width and height should be no smaller than 48.
            E.g. `(200, 200, 3)` would be one valid value.
        pooling: Optional pooling mode for feature extraction
            when `include_top` is `False`.
            - `None` means that the output of the model will be
                the 4D tensor output of the
                last convolutional layer.
            - `avg` means that global average pooling
                will be applied to the output of the
                last convolutional layer, and thus
                the output of the model will be a 2D tensor.
            - `max` means that global max pooling will
                be applied.
        classes: optional number of classes to classify images
            into, only to be specified if `include_top` is True, and
            if no `weights` argument is specified.
    # Returns
        A Keras model instance.
    # Raises
        ValueError: in case of invalid argument for `weights`,
            or invalid input shape.
    """
    if weights not in {'imagenet', None}:
        raise ValueError('The `weights` argument should be either '
                         '`None` (random initialization) or `imagenet` '
                         '(pre-training on ImageNet).')

    if weights == 'imagenet' and include_top and classes != 1000:
        raise ValueError('If using `weights` as imagenet with `include_top`'
                         ' as true, `classes` should be 1000')
    # Determine proper input shape
    input_shape = _obtain_input_shape(input_shape,
                                      default_size=224,
                                      min_size=32,
                                      data_format=K.image_data_format(),
                                      include_top=include_top)

    if input_tensor is None:
        img_input = Input(shape=input_shape)
    else:
        if not K.is_keras_tensor(input_tensor):
            img_input = Input(tensor=input_tensor, shape=input_shape)
        else:
            img_input = input_tensor
    # Block 1
    x = Conv2D(filter_size, (3, 3), activation='relu', padding='same', name='block1_conv1')(img_input)
    x = Conv2D(filter_size, (3, 3), activation='relu', padding='same', name='block1_conv2')(x)
    x = SpatialDropout2D(rate=dropout) (x)
    x = MaxPooling2D((2, 2), name='block1_pool')(x)

    # Block 2
    x = Conv2D(filter_size*2, (3, 3), activation='relu', padding='same', name='block2_conv1')(x)
    dn_feat1 = x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv2')(x)
    x = SpatialDropout2D(rate=dropout) (x)
    x = MaxPooling2D((2, 2), name='block2_pool')(x)

    # Block 3
    x = Conv2D(filter_size*4, (3, 3), activation='relu', padding='same', name='block3_conv1')(x)
    x = Conv2D(filter_size*4, (3, 3), activation='relu', padding='same', name='block3_conv2')(x)
    x = Conv2D(filter_size*4, (3, 3), activation='relu', padding='same', name='block3_conv3')(x)
    dn_feat2 = x = Conv2D(filter_size*4, (3, 3), dilation_rate=4, activation='relu', padding='same', name='block3_conv4')(x)
    x = Dropout(rate=dropout) (x)
    x = MaxPooling2D((2, 2), name='block3_pool')(x)

    # Block 4
    x = Conv2D(filter_size*8, (3, 3), activation='relu', padding='same', name='block4_conv1')(x)
    x = Conv2D(filter_size*8, (3, 3), activation='relu', padding='same', name='block4_conv2')(x)
    x = Conv2D(filter_size*8, (3, 3), activation='relu', padding='same', name='block4_conv3')(x)
    dn_feat3 = x = Conv2D(filter_size*8, (3, 3), activation='relu', padding='same', name='block4_conv4')(x)
    x = Dropout(rate=dropout) (x)
    x = MaxPooling2D((2, 2), name='block4_pool')(x)

    # Block 5
    x = Conv2D(filter_size*8, (3, 3), activation='relu', padding='same', name='block5_conv1')(x)
    x = Conv2D(filter_size*4, (3, 3), activation='relu', padding='same', name='block5_conv2')(x)
    x = Conv2D(filter_size*4, (3, 3), activation='relu', padding='same', name='block5_conv3')(x)
    dn_feat4 = x = Conv2D(filter_size*2, (3, 3), activation='sigmoid', padding='same', name='block5_conv4')(x)
    x = Dropout(rate=dropout) (x)
    x = MaxPooling2D((2, 2), name='block5_pool')(x)

    # Ensure that the model takes into account
    # any potential predecessors of `input_tensor`.
    if input_tensor is not None:
        inputs = get_source_inputs(input_tensor)
    else:
        inputs = img_input

    if weights == None and classes != 1000:
        output_shape = Model(inputs, x).output_shape
        if pooling == 'avg':
            x = AveragePooling2D((output_shape[1], output_shape[2]))(x)
        elif pooling == 'max':
            x = MaxPooling2D((output_shape[1], output_shape[2]))(x)
        x = Conv2D(classes, (1, 1), activation='softmax', padding='same', name='output') (x)

    elif include_top and classes == 1000:
        # Classification block
        x = Flatten(name='flatten')(x)
        x = Dense(4096, activation='relu', name='fc1')(x)
        x = Dense(4096, activation='relu', name='fc2')(x)
        x = Dense(classes, activation='softmax', name='predictions')(x)
    else:
        if pooling == 'avg':
            x = GlobalAveragePooling2D()(x)
        elif pooling == 'max':
            x = GlobalMaxPooling2D()(x)

    # Create model.
    model = Model(inputs, x, name='vgg19')

    # load weights
    if weights == 'imagenet':
        if include_top:
            weights_path = get_file('vgg19_weights_tf_dim_ordering_tf_kernels.h5',
                                    WEIGHTS_PATH,
                                    cache_subdir='models')
        else:
            weights_path = get_file('vgg19_weights_tf_dim_ordering_tf_kernels_notop.h5',
                                    WEIGHTS_PATH_NO_TOP,
                                    cache_subdir='models')
        model.load_weights(weights_path)
        if K.backend() == 'theano':
            layer_utils.convert_all_kernels_in_model(model)

        if K.image_data_format() == 'channels_first':
            if include_top:
                maxpool = model.get_layer(name='block5_pool')
                shape = maxpool.output_shape[1:]
                dense = model.get_layer(name='fc1')
                layer_utils.convert_dense_weights_data_format(dense, shape, 'channels_first')

            if K.backend() == 'tensorflow':
                warnings.warn('You are using the TensorFlow backend, yet you '
                              'are using the Theano '
                              'image data format convention '
                              '(`image_data_format="channels_first"`). '
                              'For best performance, set '
                              '`image_data_format="channels_last"` in '
                              'your Keras config '
                              'at ~/.keras/keras.json.')

    model.input_tensor_fcn = inputs
    model.tensor_hooks_fcn = [dn_feat1, dn_feat2, dn_feat3, dn_feat4, x]
    return model



if __name__ == '__main__':

    model = VGG19(include_top=False, weights='imagenet', input_shape=(224*1, 224*1, 3))
    img_path = 'doggos.jpg'
    img = image.load_img(img_path, target_size=(224*1, 224*1))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    print('Input image shape:', x.shape)

    preds = model.predict(x)
    dd = preds[0].max(axis=2)

    import matplotlib.pyplot as plt
    plt.figure(); plt.subplot(121); plt.imshow(x[0]); plt.subplot(122); plt.imshow(dd); 
    plt.savefig('dog_hmap.png', bbox_inches='tight')
