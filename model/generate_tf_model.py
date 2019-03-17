"""
This file generates data from a given model, as set in the first line after the imports.
It is important that all structural parameters, if changed in the model generation,
also need to be changed here
"""

import os
from copy import deepcopy

import numpy as np
# run this on CPU since (in my case) I couldn't train and predict on the GPU
# at the same time
from tensorflow.python.framework.errors_impl import InvalidArgumentError

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
import tensorflow as tf
from tensorflow.python.keras.backend import set_session
from tensorflow._api.v1.keras.layers import Input, LSTM, Dense, concatenate, Masking
from tensorflow._api.v1.keras.models import Model
from tensorflow._api.v1.keras.optimizers import Adam
from tensorflow._api.v1.keras.preprocessing.sequence import pad_sequences
import model.make_tf_structure as make_tf
from tensorflow._api.v1.keras.utils import to_categorical
import settings.constants as c

weights_filename = "data/weights_tf/weights-improvement-tf-project-33-0.0411.hdf5"

config = tf.ConfigProto()
config.gpu_options.allow_growth = True  # dynamically grow the memory used on the GPU
sess = tf.Session(config=config)
set_session(sess)  # set this TensorFlow session as the default session for Keras

# our input are three sequences, which (zipped) represent a melody:
# the pitch, the lentgh of a note, and it's offset.
# It is important to note that the offset is only needed for allowing the melody
# to more easily stay in the beat
pitch_input = Input(shape=(c.sequence_length, 37), dtype='float32', name='pitch_input')
length_input = Input(shape=(c.sequence_length, 16), dtype='float32', name='length_input')
offset_input = Input(shape=(c.sequence_length, 4), dtype='float32', name='offset_input')

# a concatenation layer
concatenated_input = concatenate([pitch_input, length_input, offset_input], axis=-1)

# masking removes dummy values that were introduced to train on the first notes in a melody
masked_input = Masking(0.0)(concatenated_input)

# a normal LSTM layer with 512 nodes
lstm_layer = LSTM(512)(masked_input)

# two dense layers as output layers, applying the softmax activation function
pitch_output = Dense(37, activation='softmax', name='pitch_output')(lstm_layer)
length_output = Dense(16, activation='softmax', name='length_output')(lstm_layer)

# here we define our model
model = Model(inputs=[pitch_input, length_input, offset_input],
              outputs=[pitch_output, length_output])

# we load the weights we have saved above
try:
    model.load_weights(weights_filename)
except InvalidArgumentError:
    print("For generating a melody, you need to specify a correct weight file above")
    import sys

    sys.exit(1)

# Model is compiled using the Adam Optimizer
model.compile(loss={'pitch_output': 'categorical_crossentropy',
                    'length_output': 'categorical_crossentropy'},
              optimizer=Adam())

########################################################
number_of_melodies_to_generate = 1

headache = [(71.0, 1.0, 0.0), (76.0, 1.0, 1.0), (69.0, 2.0, 2.0), (74.0, 0.5, 4.0),
            (69.0, 0.5, 4.5), (72.0, 3.0, 5.0), (71.0, 2.0, 8.0), (64.0, 2.0, 10.0)]

# start_sequence = [(60, 1.0, 0.0)]
start_sequence = headache

for melody_nr in range(number_of_melodies_to_generate):

    start_pitches = []
    start_lengths = []
    start_offsets = []

    # we transform the input so that we can use it for one hot vectors
    for pitch, length, offset in start_sequence:
        start_pitches.append(make_tf.pitch_to_int(pitch))
        start_lengths.append(make_tf.length_to_int(1.0))
        start_offsets.append(make_tf.offset_to_binary_array(offset))

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True

    music_info_list = deepcopy(start_sequence)

    # we calculate the one hot vectors and pad every sequence in the front
    # so that we can also start with an empty or almost empty start-sequence
    pitch_input_one_hot = to_categorical(start_pitches, num_classes=37)
    pitch_input_pad = pad_sequences([pitch_input_one_hot], maxlen=c.sequence_length,
                                    dtype='float32', padding='pre',
                                    truncating='post', value=0.0)

    length_input_one_hot = to_categorical(start_lengths, num_classes=16)
    length_input_pad = pad_sequences([length_input_one_hot], maxlen=c.sequence_length,
                                     dtype='float32', padding='pre',
                                     truncating='post', value=0.0)

    offset_input_pad = pad_sequences([start_offsets], maxlen=c.sequence_length,
                                     dtype='float32', padding='pre',
                                     truncating='post', value=0.0)

    for i in range(100):
        # this returns the prediction as a probability distribution
        pitch_pred, length_pred = model.predict([pitch_input_pad, length_input_pad, offset_input_pad], verbose=0)

        # with simple argmax, we would always get the same output, but like that the melody creation is
        # randomized and more "creative"
        pitch_idx = np.random.choice(a=len(pitch_pred[0]), size=1, p=pitch_pred[0])[0]
        length_idx = np.random.choice(a=len(length_pred[0]), size=1, p=length_pred[0])[0]
        # offset plus length of last note
        offset = music_info_list[-1][2] + music_info_list[-1][1]

        # save the result in our list
        music_info_list.append((make_tf.int_to_pitch(pitch_idx),
                                make_tf.int_to_length(length_idx),
                                offset))

        # we append the one hot vector of our predictions and delete the first entry of our
        # sequence so far
        pitch_one_hot = to_categorical([pitch_idx], num_classes=37)
        length_one_hot = to_categorical([length_idx], num_classes=16)
        offset_bool = make_tf.offset_to_binary_array(offset)
        offset_bool = np.reshape(offset_bool, (1, 4))

        pitch_input_res = np.reshape(pitch_input_pad, (30, 37))
        pitch_input_res = np.concatenate([pitch_input_res, pitch_one_hot], axis=0)
        pitch_input_res = pitch_input_res[1:]
        pitch_input_pad = np.reshape(pitch_input_res, (1, 30, 37))

        length_input_res = np.reshape(length_input_pad, (30, 16))
        length_input_res = np.concatenate([length_input_res, length_one_hot], axis=0)
        length_input_res = length_input_res[1:]
        length_input_pad = np.reshape(length_input_res, (1, 30, 16))

        offset_input_res = np.reshape(offset_input_pad, (30, 4))
        offset_input_res = np.concatenate([offset_input_res, offset_bool], axis=0)
        offset_input_res = offset_input_res[1:]
        offset_input_pad = np.reshape(offset_input_res, (1, 30, 4))

    # show the model in musescore 2. If you don't have musescore, comment this out
    # and maybe just print the music_info_list
    make_tf.tf_model_output_to_musescore(music_info_list)
