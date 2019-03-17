from copy import deepcopy
from statistics import mean, stdev

import music_utils.simple_classes as simple
import settings.constants as c


def simple_skyline_algorithm_from_simple(song_or_part_or_parts_list,
                                         split: bool, max_rest: float = 4.0,
                                         min_melody_length: float = 16.0) -> [(float, simple.NoteList)]:
    """
    a simple skyline algorithm, as described in the paper
    :param song_or_part_or_parts_list:
    :param split:
    :param max_rest:
    :param min_melody_length:
    :return:
    """
    if type(song_or_part_or_parts_list) == simple.Part or \
            type(song_or_part_or_parts_list) == simple.Song:
        notes = song_or_part_or_parts_list.notes()

    elif type(song_or_part_or_parts_list) == list:
        notes = simple.NoteList()
        for part_or_song in song_or_part_or_parts_list:
            assert type(part_or_song) == simple.Part or type(part_or_song) == simple.Song or \
                   type(part_or_song) == simple.NoteList

            if type(part_or_song) == simple.NoteList:
                notes.extend(deepcopy(part_or_song))
            else:
                notes.extend(part_or_song.notes())

        notes.sort()

    elif type(song_or_part_or_parts_list) == simple.NoteList:
        notes = deepcopy(song_or_part_or_parts_list)

    else:
        raise ValueError("Type {t} doesn't fit, must be either simple.Part, simple.Song,"
                         "simple.NoteList or a list with the above!".format(t=type(song_or_part_or_parts_list)))

    assert type(notes) == simple.NoteList

    current_note = None
    current_end = -1

    melody = simple.NoteList()

    for note in notes:
        pitch = note.pitch

        # just allow pitches in the specified range
        # also immediately removes rests, which have per definition a pitch of 200
        # this is because protocol buffers have problems handling negative ints efficiently
        if not (c.music_settings.min_pitch <= pitch <= c.music_settings.max_pitch):
            continue

        # no jumps down greater than an octave!
        if current_note:
            if current_note.pitch - pitch > 12:
                continue

        if current_end <= note.offset:
            if current_note is not None:
                melody.append(current_note)
            current_note = note
            current_end = note.end()

        elif pitch > current_note.pitch or (note.offset > current_note.offset and
                                            note.part == current_note.part):
            if note.offset - current_note.offset > 0:
                current_note.length = note.offset - current_note.offset
                melody.append(current_note)
            current_note = note
            current_end = note.end()

    if current_note is not None:
        melody.append(current_note)

    assert is_sequence(melody)

    if not split:
        return [(0.0, melody)]

    return make_full_sub_melodies(melody, max_rest=max_rest,
                                  min_melody_length=min_melody_length)


def tf_skyline(song: simple.Song, split: bool,
               max_rest: float = 4.0, max_melody_length: float = 16.0) -> [(float, simple.NoteList)]:
    """
    an upgrade to skyline algorithm discarding improbable parts
    :param song:
    :param split:
    :param max_rest:
    :param max_melody_length:
    :return:
    """
    parts = []
    average_volumes = []
    average_pitches = []
    all_notes = []

    for part in song.parts:
        temp_part_notes = simple_skyline_algorithm_from_simple(part, split=False)[0][1]
        if len(temp_part_notes) > 0:
            parts.append(simple_skyline_algorithm_from_simple(part, split=False)[0][1])

    for i, part in enumerate(parts):
        assert is_sequence(part)
        part: simple.NoteList
        average_volumes.append(mean([note.volume for note in part]))
        average_pitches.append(mean([note.pitch for note in part]))
        all_notes.extend(part)

    # for calculating variance, at least 2 notes are needed. melody_length//4 is a
    # lower bound for the least smallest amount of notes
    if len(all_notes) < max(2, int(max_melody_length // 4)):
        return None

    probable_melody_parts = []

    mean_volume = mean([n.volume for n in all_notes])
    stdev_volume = stdev([n.volume for n in all_notes])
    mean_pitch = mean([n.pitch for n in all_notes])
    stdev_pitch = stdev([n.pitch for n in all_notes])

    for avg_vol, avg_pitch, part in zip(average_volumes, average_pitches, parts):
        if avg_vol >= mean_volume - stdev_volume:
            probable_melody_parts.append(part)
        elif avg_pitch >= mean_pitch - stdev_pitch:
            probable_melody_parts.append(part)

    result = simple_skyline_algorithm_from_simple(probable_melody_parts, split,
                                                  max_rest, max_melody_length)

    return result


def is_sequence(notes: simple.NoteList):
    """
    returns if notes are sequential, True if no notes are overlapping
    :param notes:
    :return:
    """
    return all([first.end() <= second.offset for first, second in zip(notes[:-1], notes[1:])])


def make_full_sub_melodies(melody: simple.NoteList, max_rest: float, min_melody_length: float) \
        -> [(float, simple.NoteList)]:
    """
    splits the melody at rests longer than max_rest with split_melody, than calls make_breaks_and_start
    :param melody:
    :param max_rest:
    :param min_melody_length:
    :return:
    """
    if not melody:
        return None
    melodies = split_melody(melody, max_rest, min_melody_length)
    full_melodies = []
    for m in melodies:
        full_melodies.append(make_breaks_and_start(m))

    return full_melodies


def split_melody(melody: simple.NoteList, max_rest: float, min_melody_length: float):
    """
    splits the melody at rests longer than max_rests
    :param melody:
    :param max_rest:
    :param min_melody_length:
    :return:
    """
    if not melody:
        return None

    split_indexes = [0]
    split_indexes.extend([i + 1
                          for i, (n1, n2)
                          in enumerate(zip(melody[:-1], melody[1:]))
                          if n2.offset > n1.end() + max_rest])
    split_indexes.append(len(melody))

    # splits the melody at breaks that take more than max_rest beats
    melodies = [simple.NoteList(melody[i1:i2])
                for i1, i2
                in zip(split_indexes[:-1], split_indexes[1:])
                if melody[i1].end() + min_melody_length <= melody[i2 - 1].offset]
    # the melodies should be at least min_melody_length beats long!

    return melodies


def make_breaks_and_start(melody: simple.NoteList) -> (float, simple.NoteList):
    """
    given a note list, returns a new note list with the rests before and after the melody starts deleted,
    and rests inserted at places where they belong
    :param melody: the NoteList specifying the melody
    :return: a new NoteList
    """
    start = melody[0].offset
    first_measure_start = (start // 4) * 4

    copied_melody = deepcopy(melody)
    for note in copied_melody:
        note.offset -= first_measure_start

    full_list = simple.NoteList()

    if copied_melody[0].offset > 0.0:
        full_list.append(simple.Note(offset=0.0,
                                     length=copied_melody[0].offset,
                                     pitch=200,
                                     volume=0,
                                     part=copied_melody[0].part))

    full_list.append(copied_melody[0])

    for note in copied_melody[1:]:
        if note.offset > full_list[-1].end():
            full_list.append(simple.Note(offset=full_list[-1].end(),
                                         length=note.offset - full_list[-1].end(),
                                         pitch=200,
                                         volume=0,
                                         part=full_list[-1].part))

        full_list.append(note)

    return first_measure_start, full_list


if __name__ == '__main__':

    import os
    from settings.music_info_pb2 import VanillaStreamPB
    from random import shuffle

    file_list = []

    for dirname, _, filenames in os.walk(c.MXL_DATA_FOLDER):

        for filename in filenames:
            if filename.endswith('.pb'):
                file_list.append(os.path.join(dirname, filename))
                pass

    shuffle(file_list)

    for filename in file_list[0:1]:

        assert filename.endswith(".pb")

        proto_buffer_path = filename

        print(filename)

        with open(proto_buffer_path, 'rb') as fp:
            proto_buffer = VanillaStreamPB()
            proto_buffer.ParseFromString(fp.read())
            assert proto_buffer.HasField('info')

        simple_song = simple.Song(proto_buffer)

        print(simple_song)

        # full_stream = stream_from_pb(proto_buffer)
        simple_melody = simple_skyline_algorithm_from_simple(simple_song, split=False)[0][1]
        # print(json_format.MessageToJson(proto_buffer.info))

        tf_melody = tf_skyline(simple_song, split=False)[0][1]

        all_melodies_song = simple.Song(list_of_parts_or_note_lists=[simple_melody,
                                                                     tf_melody])

        # all_melodies_song.m21_stream().show('midi')

        full_melodies = make_full_sub_melodies(tf_melody, max_rest=4.0, min_melody_length=16.0)

        print("Full melodies:")

        [print(m) for m in full_melodies]
