import tensorflow
from keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D, SeparableConv2D, UpSampling2D, BatchNormalization, Input, GlobalAveragePooling2D

from keras.models import Sequential
from keras.models import Model
import tensorflow as tf

def entry_flow(inputs) :
    
    # channel-1 with kernel size of 3
    
    x = Conv2D(32, 3, strides = 2, padding='same')(inputs)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = Conv2D(64,3,padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    previous_block_activation_x = x

    for size in [128, 256, 728] :

        x = Activation('relu')(x)
        x = SeparableConv2D(size, 3, padding='same')(x)
        x = BatchNormalization()(x)

        x = Activation('relu')(x)
        x = SeparableConv2D(size, 3, padding='same')(x)
        x = BatchNormalization()(x)

        x = MaxPooling2D(3, strides=2, padding='same')(x)

        residual_x = Conv2D(size, 1, strides=2, padding='same')(previous_block_activation_x)

        x = tensorflow.keras.layers.Add()([x, residual_x])
        previous_block_activation_x = x

    # channel-2 with kernel size of 7

    y = Conv2D(32, 7, strides = 2, padding='same')(inputs)
    y = BatchNormalization()(y)
    y = Activation('relu')(y)

    y = Conv2D(64,7,padding='same')(y)
    y = BatchNormalization()(y)
    y = Activation('relu')(y)

    previous_block_activation_y = y

    for size in [128, 256, 728] :

        y = Activation('relu')(y)
        y = SeparableConv2D(size, 3, padding='same')(y)
        y = BatchNormalization()(y)

        y = Activation('relu')(y)
        y = SeparableConv2D(size, 3, padding='same')(y)
        y = BatchNormalization()(y)

        y = MaxPooling2D(3, strides=2, padding='same')(y)

        residual_y = Conv2D(size, 1, strides=2, padding='same')(previous_block_activation_y)

        y = tensorflow.keras.layers.Add()([y, residual_y])
        previous_block_activation_y = y

    return x, y


def middle_flow(x, y, num_blocks=8) :

    previous_block_activation_x = x

    for _ in range(num_blocks) :

        x = Activation('relu')(x)
        x = SeparableConv2D(728, 3, padding='same')(x)
        x = BatchNormalization()(x)

        x = Activation('relu')(x)
        x = SeparableConv2D(728, 3, padding='same')(x)
        x = BatchNormalization()(x)

        x = Activation('relu')(x)
        x = SeparableConv2D(728, 3, padding='same')(x)
        x = BatchNormalization()(x)

        x = tensorflow.keras.layers.Add()([x, previous_block_activation_x])
        previous_block_activation_x = x
        
    previous_block_activation_y = y

    for _ in range(num_blocks) :

        y = Activation('relu')(y)
        y = SeparableConv2D(728, 3, padding='same')(y)
        y = BatchNormalization()(y)

        y = Activation('relu')(y)
        y = SeparableConv2D(728, 3, padding='same')(y)
        y = BatchNormalization()(y)

        y = Activation('relu')(y)
        y = SeparableConv2D(728, 3, padding='same')(y)
        y = BatchNormalization()(y)

        y = tensorflow.keras.layers.Add()([y, previous_block_activation_y])
        previous_block_activation_y = y


    return x, y

def exit_flow(x, y) :

    previous_block_activation_x = x

    x = Activation('relu')(x)
    x = SeparableConv2D(728, 3, padding='same')(x)
    x = BatchNormalization()(x)

    x = Activation('relu')(x)
    x = SeparableConv2D(1024, 3, padding='same')(x) 
    x = BatchNormalization()(x)

    x = MaxPooling2D(3, strides=2, padding='same')(x)

    residual_x = Conv2D(1024, 1, strides=2, padding='same')(previous_block_activation_x)
    x = tensorflow.keras.layers.Add()([x, residual_x])

    x = Activation('relu')(x)
    x = SeparableConv2D(728, 3, padding='same')(x)
    x = BatchNormalization()(x)

    x = Activation('relu')(x)
    x = SeparableConv2D(1024, 3, padding='same')(x)
    x = BatchNormalization()(x)
    
    previous_block_activation_y = y

    y = Activation('relu')(y)
    y = SeparableConv2D(728, 3, padding='same')(y)
    y = BatchNormalization()(y)

    y = Activation('relu')(y)
    y = SeparableConv2D(1024, 3, padding='same')(y) 
    y = BatchNormalization()(y)

    y = MaxPooling2D(3, strides=2, padding='same')(y)

    residual_y = Conv2D(1024, 1, strides=2, padding='same')(previous_block_activation_y)
    y = tensorflow.keras.layers.Add()([y, residual_y])

    y = Activation('relu')(y)
    y = SeparableConv2D(728, 3, padding='same')(y)
    y = BatchNormalization()(y)

    y = Activation('relu')(y)
    y = SeparableConv2D(1024, 3, padding='same')(y)
    y = BatchNormalization()(y)
    
    # fuse channel-1 and channel-2
    z = tf.math.add(x,y)
    z = GlobalAveragePooling2D()(z)
    z = Dense(2, activation='sigmoid')(z)

    return z

