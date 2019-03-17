import os
import sys
import traceback

import settings.constants as c
import settings.music_info_pb2 as music_info
from music_utils.vanilla_part import VanillaPart
from music_utils.vanilla_stream import VanillaStream


def make_piece_of_music_pb(m21_stream: VanillaStream, error_message: str, update_pb=True):
    """
    makes a new protocol buffer entry to the protocol buffer specified in settings.constants,
    based on the m21_stream file
    :param m21_stream: Music piece that should be saved
    :param error_message: an optional message specifying why this file is not valid for the current settings
    :return:
    """
    music_file = music_info.PieceOfMusic(filepath=os.path.relpath(m21_stream.id, c.MXL_DATA_FOLDER),
                                         valid=m21_stream.valid)
    if m21_stream.min_metronome:
        music_file.min_metronome = int(m21_stream.min_metronome)
    if m21_stream.max_metronome:
        music_file.max_metronome = int(m21_stream.max_metronome)
    if error_message:
        music_file.error = getattr(music_info, error_message)
    else:

        assert m21_stream.valid

        music_file.key_correlation = m21_stream.key_correlation

        for p in m21_stream.parts:
            p: VanillaPart

            temp_part = music_file.parts.add(name=p.partName)
            temp_part.average_pitch = p.average_pitch
            temp_part.average_volume = p.average_volume

            try:
                temp_part.key_correlation = p.key_by_name(m21_stream.key).correlationCoefficient
            except ValueError:
                temp_part.key_correlation = -1.0

            temp_part.note_percentage = p.note_percentage
            temp_part.lyrics_percentage = p.lyrics_percentage

    if update_pb:
        c.music_protocol_buffer.counter = c.music_protocol_buffer.counter + 1
        c.music_protocol_buffer.music_data.add().CopyFrom(music_file)

    return music_file


def put_in_protocol_buffer(m21_stream: VanillaStream, error_message: str = None):
    """
    adds the information of the m21_stream file to the protocol buffer specified by the current settings.
    For further information, see settings.constants
    :param m21_stream: a VanillaStream instance of our musical piece
    :param error_message: an optional message saying why the stream is invalid with the current settings
    :return:
    """
    c.music_info_dict_lock.acquire()
    try:
        if proto_buffer_entry_exists(m21_stream.id)[0]:
            c.music_info_dict_lock.release()
            return

        piece_of_music_pb = make_piece_of_music_pb(m21_stream, error_message)
        c.existing_files[m21_stream.id] = m21_stream.valid

        if c.music_protocol_buffer.counter >= c.UPDATE_FREQUENCY:

            c.music_protocol_buffer.counter = 0
            serialized_byte_stream = c.music_protocol_buffer.SerializeToString()
            c.music_info_dict_lock.release()

            c.music_info_file_lock.acquire()
            print("\rcurrently writing protocol buffer\t\t\t\t\t", file=sys.stderr, end='', flush=True)
            with open(c.PROTOCOL_BUFFER_LOCATION, 'wb') as fp:
                fp.write(serialized_byte_stream)
            print("\rfinished writing protocol buffer \t\t\t\t\t", file=sys.stderr, end='', flush=True)
            c.music_info_file_lock.release()

        else:
            c.music_info_dict_lock.release()

        return piece_of_music_pb

    except:
        traceback.print_exc()
        try:
            c.music_info_dict_lock.release()
        except RuntimeError:
            pass
        try:
            c.music_info_file_lock.release()
        except RuntimeError:
            pass


def make_invalid_in_protocol_buffer(filename, error):
    try:
        c.music_info_dict_lock.acquire()
        for file in c.music_protocol_buffer.music_data:
            if filename == os.path.join(c.MXL_DATA_FOLDER, file.filepath):
                file.valid = False
                file.ClearField('key')
                file.ClearField('key_correlation')
                file.ClearField('parts')
                file.error = getattr(music_info, error)
                break

    finally:
        c.music_info_dict_lock.release()


def proto_buffer_entry_exists(filename: str) -> (bool, bool):
    """
    tests if the file is already in our database for the current settings
    :param filename: name of the (complete) filepath
    :return: True if the entry exists, False if it doesn't
    """

    relative_filename = os.path.relpath(filename, c.MXL_DATA_FOLDER)
    valid = False
    exists = relative_filename in c.existing_files
    if exists:
        valid = c.existing_files[relative_filename]

    return exists, valid


def save_vanilla_stream_pb(m21_stream: VanillaStream, info: music_info.PieceOfMusic):
    new_file_path = m21_stream.id.replace('.mxl', '.pb')

    # if in some other program or place this was already created
    if os.path.exists(new_file_path):
        return

    if not info:
        info = make_piece_of_music_pb(m21_stream, "", update_pb=False)

    proto_buffer = _make_vanilla_stream_proto_buffer(m21_stream, info)

    with open(new_file_path, 'xb') as fp:
        fp.write(proto_buffer.SerializeToString())


def _make_vanilla_stream_proto_buffer(m21_stream: VanillaStream,
                                      temp_info: music_info.PieceOfMusic) -> music_info.VanillaStreamPB:
    """
    makes a proto buffer file from a vanilla stream, that is small and fast :D
    :param m21_stream:
    :param temp_info:
    :return:
    """
    proto_buffer = music_info.VanillaStreamPB()
    proto_buffer.filepath = os.path.relpath(m21_stream.id, c.MXL_DATA_FOLDER)

    proto_buffer.info.CopyFrom(temp_info)

    for p in m21_stream.parts:

        proto_part = proto_buffer.parts.add(name=p.partName)

        offsets = []
        lengths = []
        pitches = []
        volumes = []

        for note in p.flat.getElementsByClass(('Note', 'Rest')):
            offsets.append(note.offset)
            lengths.append(note.quarterLength)
            if note.isNote:
                pitches.append(int(note.pitch.ps))
                volumes.append(int(note.volume.velocity))
            elif note.isRest:
                pitches.append(200)
                volumes.append(0)

        assert len(offsets) == len(lengths) == len(pitches) == len(volumes)

        proto_part.offsets.extend(offsets)
        proto_part.lengths.extend(lengths)
        proto_part.pitches.extend(pitches)
        proto_part.volumes.extend(volumes)

    return proto_buffer
