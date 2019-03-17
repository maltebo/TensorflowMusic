import music21 as m21
import numpy as np
from tensorflow._api.v1.keras.preprocessing.sequence import pad_sequences
from tensorflow._api.v1.keras.utils import to_categorical

import settings.constants as c
import settings.music_info_pb2 as music_info
from music_utils.vanilla_stream import VanillaStream


def make_tf_data(settings=c.music_settings):
    """
    calculates all the necessary information for our training and returns it.
    reads the information from protocol buffers created in the preprocessing step
    :param settings: setting information for the melodies you're looking at
    :return:
    """

    pitches_input = []
    lengths_input = []
    offsets_input = []
    pitches_output = []
    lengths_output = []

    min_sequence_length = c.sequence_length + 1  # number of min beats per melody

    while not c.melody_work_queue.empty():
    # how many of the songs created do you want to use for training?
    # all is too much for my computer :D
    # for i in range(100):

        melody = c.melody_work_queue.get()

        # that's the kind of melody we want to see
        if not melody.endswith('_tf_skyline.melody_pb'):
            continue

        with open(melody, 'rb') as fp:
            melody_list = music_info.MelodyList()
            melody_list.ParseFromString(fp.read())

        for m in melody_list.melodies:
            if len(m.lengths) < min_sequence_length:
                continue

            # append one-hot encoding/binary encoding of offset
            pitches = [to_categorical([pitch_to_int(p, settings)], num_classes=37) for p in m.pitches]
            lengths = [to_categorical([length_to_int(l)], num_classes=16) for l in m.lengths]
            offsets = [offset_to_binary_array(o) for o in m.offsets]

            for time_step in range(len(m.lengths) - 2):
                pitches_input.append(pitches[max(0, time_step - c.sequence_length + 1): time_step + 1])
                lengths_input.append(lengths[max(0, time_step - c.sequence_length + 1): time_step + 1])
                offsets_input.append(offsets[max(0, time_step - c.sequence_length + 1): time_step + 1])

                pitches_output.append(pitches[time_step + 1])
                lengths_output.append(lengths[time_step + 1])

    # pad all sequences with zeros / zero arrays until max sequence length specified in constants
    # is reached
    pitches_input = pad_sequences(pitches_input, maxlen=c.sequence_length, dtype='float32')
    lengths_input = pad_sequences(lengths_input, maxlen=c.sequence_length, dtype='float32')
    offsets_input = pad_sequences(offsets_input, maxlen=c.sequence_length, dtype='float32')

    # we need to do some reshaping, yey
    pitches_input = np.reshape(pitches_input, (-1, c.sequence_length, 37))
    lengths_input = np.reshape(lengths_input, (-1, c.sequence_length, 16))
    offsets_input = np.reshape(offsets_input, (-1, c.sequence_length, 4))

    pitches_output = np.reshape(pitches_output, (-1, 37))
    lengths_output = np.reshape(lengths_output, (-1, 16))

    return pitches_input, lengths_input, offsets_input, pitches_output, lengths_output


def pitch_to_int(pitch, settings=c.music_settings):
    """
    turns a pitch into its corresponding one hot index
    :param pitch: the pitch (60 = C4)
    :param settings: a settings file saying what should be used as min and max pitch
    :return:
    """
    assert (settings.min_pitch <= pitch <= settings.max_pitch) or pitch == 200
    return int((pitch - settings.min_pitch + 1) % (200 - settings.min_pitch + 1))


def int_to_pitch(int_pitch, settings=c.music_settings):
    """
    turns an index in a one hot vector to the corresponding note pitch
    :param int_pitch:
    :param settings:
    :return:
    """
    assert int_pitch <= settings.max_pitch - settings.min_pitch + 1
    if int_pitch == 0:
        return 200
    return int_pitch + settings.min_pitch - 1


def length_to_int(length):
    """
    turns a note length into its integer one hot index
    :param length:
    :return:
    """
    assert 0.25 <= length <= 4.0
    return int(length * 4) - 1


def int_to_length(int_length):
    """
    turns an integer index value into a note length
    :param int_length:
    :return:
    """
    assert 0 <= int_length <= 15
    return (int_length + 1) / 4.0


def offset_to_binary_array(offset):
    """
    turns an offset into the binary representation, e.g. 3 => [0, 0, 1, 1],
    ignoring all offsets outside of a specific measure
    :param offset:
    :return:
    """
    return np.asarray([int(x) for x in format(int((offset % 4) * 4), '04b')[:]], dtype='float32')


def tf_model_output_to_musescore(music_info_list):
    """
    turns a music info list as saved in generate_tf_model into a VanillaStream and shows it in Musescore
    :param music_info_list:
    :return:
    """
    vs = VanillaStream()

    for pitch, length, offset in music_info_list:
        if pitch == 200:
            n = m21.note.Rest()
        else:
            n = m21.note.Note()
            n.pitch.ps = pitch
        n.quarterLength = length
        n.offset = offset
        vs.insert(n)

    # from my experience, midi works best as keyword, musicxml has lots of bugs
    vs.show('midi')
