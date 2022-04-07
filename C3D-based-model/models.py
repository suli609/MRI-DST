"""
A collection of models we'll use to attempt to classify videos.
"""
from keras.layers import Dense, Flatten, Dropout, ZeroPadding3D,Activation, SpatialDropout3D,TimeDistributed,GlobalAveragePooling1D,GlobalMaxPooling2D,Input,concatenate
from tensorflow.layers import batch_normalization
from keras.layers.normalization import BatchNormalization
import tensorflow as tf
from keras.layers.recurrent import LSTM
from keras.models import Sequential, load_model, Model
from keras.optimizers import Adam, RMSprop
from keras.layers.wrappers import TimeDistributed
from keras.layers import (Conv2D, MaxPooling3D, Conv3D,
    MaxPooling2D)
from collections import deque
import sys

class ResearchModels():
    def __init__(self, nb_classes, model, seq_length,
                 saved_model=None, features_length=2048,input_shape_num=6):
        
        # Set defaults.
        self.seq_length = seq_length
        self.load_model = load_model
        self.saved_model = saved_model
        self.nb_classes = nb_classes
        self.feature_queue = deque()
        self.input_shape_num = input_shape_num

        # Set the metrics. Only use top k if there's a need.
        metrics = ['accuracy']
        if self.nb_classes >= 10:
            metrics.append('top_k_categorical_accuracy')

        # Get the appropriate model.
        if self.saved_model is not None:
            print("Loading model %s" % self.saved_model)
            self.model = load_model(self.saved_model)
        elif model == 'c3d':
            print("Loading C3D")
        #    self.input_shape = (seq_length, 784, 784, 3)
            self.input_shape = (seq_length, 392, 392, 3)
            self.model = self.c3d()
        else:
            print("Unknown network.")
            sys.exit()
        # Now compile the network.
        optimizer = Adam(lr=1e-5, decay=1e-6)
        self.model.compile(loss='categorical_crossentropy', optimizer=optimizer,
                           metrics=metrics)

        print(self.model.summary())  

    def c3d(self):
        
        Input_seq = Input(shape = self.input_shape,)
        Input_num = Input(shape = (self.input_shape_num,),) 
        
        #1
        x = Conv3D(64, 3, 3, 3, activation='relu',
                         border_mode='same', name='conv1',
                         subsample=(1, 1, 1))(Input_seq)
        x = MaxPooling3D(pool_size=(1, 2, 2), strides=(1, 2, 2),
                               border_mode='valid', name='pool1')(x)

        x = Conv3D(128, 3, 3, 3, activation='relu',
                         border_mode='same', name='conv2',
                         subsample=(1, 1, 1))(x)
        x = MaxPooling3D(pool_size=(2, 2, 2), strides=(2, 2, 2),
                               border_mode='valid', name='pool2')(x)
        
        x = Conv3D(256, 3, 3, 3, activation='relu',
                         border_mode='same', name='conv3a',
                         subsample=(1, 1, 1))(x)
        x = Conv3D(256, 3, 3, 3, activation='relu',
                         border_mode='same', name='conv3b',
                         subsample=(1, 1, 1))(x)
        x = MaxPooling3D(pool_size=(2, 2, 2), strides=(2, 2, 2),
                               border_mode='valid', name='pool3')(x)

        x = Conv3D(512, 3, 3, 3, activation='relu',
                         border_mode='same', name='conv4a',
                         subsample=(1, 1, 1))(x)
        x = Conv3D(512, 3, 3, 3, activation='relu',
                         border_mode='same', name='conv4b',
                         subsample=(1, 1, 1))(x)
        x = MaxPooling3D(pool_size=(2, 2, 2), strides=(2, 2, 2),
                                border_mode='valid', name='pool4')(x)
        
        
        x = Conv3D(512, 3, 3, 3, activation='relu',
                         border_mode='same', name='conv5a',
                         subsample=(1, 1, 1))(x)
        x = Conv3D(512, 3, 3, 3, activation='relu',
                         border_mode='same', name='conv5b',
                         subsample=(1, 1, 1))(x)
        
        x = ZeroPadding3D(padding=(0, 1, 1))(x)
        x = MaxPooling3D(pool_size=(2, 2, 2), strides=(2, 2, 2),
                               border_mode='valid', name='pool5')(x)
        x = TimeDistributed(
                GlobalMaxPooling2D(name='global_ave1'),
                name='timeDistributed1')(x)
        x = GlobalAveragePooling1D(name='GAP2D')(x)
        x = Dense(4096, activation='relu', name='fc6')(x)
        x = Dropout(0.5)(x)
        x = Dense(4096, activation='relu', name='fc7')(x)
        x = Dropout(0.5)(x)
        x = Model(inputs=Input_seq, outputs=x )

        y = Dense(512, activation='relu', name='fc1_y')(Input_num)
        y = Dropout(0.5)(y)
        y = Model(inputs=Input_num,outputs=y)

        combined = concatenate([x.output,y.output])

        out = Dense(self.nb_classes, activation='softmax')(combined)
        model = Model(inputs=[x.input,y.input],outputs=out)
        return model

    
