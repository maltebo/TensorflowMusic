import tensorflow as tf
from tensorflow._api.v1.keras.backend import set_session
from tensorflow._api.v1.keras.layers import Input, LSTM, Dense, concatenate, Masking
from tensorflow._api.v1.keras.models import Model
from tensorflow._api.v1.keras.optimizers import Adam

config = tf.ConfigProto()
config.gpu_options.allow_growth = True  # dynamically grow the memory used on the GPU
# config.log_device_placement = True  # to log device placement (on which device the operation ran)
sess = tf.Session(config=config)
set_session(sess)

import settings.constants as c
import model.make_tf_structure as tf_struct

pitch_in, length_in, offset_in, pitch_out, length_out = tf_struct.make_tf_data()

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

model.compile(loss={'pitch_output': 'categorical_crossentropy',
                    'length_output': 'categorical_crossentropy'},
              optimizer=Adam())

# print a nice model overview
print(model.summary(90))

# save the best results automatically in files
filepath = "data/tf_weights/weights-improvement-tf-project-{epoch:02d}-{loss:.4f}.hdf5"
checkpoint = tf.keras.callbacks.ModelCheckpoint(filepath, monitor='loss', verbose=1, save_best_only=True, mode='min')

callbacks_list = [checkpoint]

# fit the model with specified number of epochs and a split of test and validation data
model.fit([pitch_in, length_in, offset_in], [pitch_out, length_out],
          epochs=20, batch_size=10, callbacks=callbacks_list, validation_split=0.2)
