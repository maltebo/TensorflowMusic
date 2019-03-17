from copy import deepcopy

import music21 as m21

import settings.constants as c
from music_utils.vanilla_part import VanillaPart
from music_utils.vanilla_stream import VanillaStream
from preprocessing.helper import FileNotFittingSettingsError
from preprocessing.helper import round_to_quarter


def make_file_container(m21_file: m21.stream.Score, m21_stream: VanillaStream):
    """
    puts only metronome marks and time signatures into a VanillaStream
    :param m21_file: the corresponding score
    :param m21_stream: the VanillaStream built from the original file
    :return:
    """
    metronome_time_sig_stream = set(m21_file.flat.getElementsByClass(('MetronomeMark',
                                                                      'TimeSignature')))

    for elem in metronome_time_sig_stream:
        elem.offset = round_to_quarter(elem.offset)
        m21_stream.insert_local(elem)


def process_file(m21_file: m21.stream.Score, m21_stream: VanillaStream):

    part_name_list = []
    number = 2

    part: m21.stream.Part
    for part in list(m21_file.parts):

        delete = False

        # delete drum channel (9) and parts with "drum" in their name
        for instr in part.getInstruments(recurse=True):
            if instr.midiChannel == 9:
                delete = True
                break
            if "drum" in instr.bestName().lower():
                delete = True
                break

        if delete:
            continue

        # needed for some instrument like obscure oboe types that are written in a different pitch system
        part.toSoundingPitch(inPlace=True)

        temp_part = VanillaPart()

        # force unique names
        if part.partName in part_name_list:
            temp_part.partName = part.partName + "_" + str(number)
            number += 1
        else:
            part_name_list.append(part.partName)
            temp_part.partName = part.partName

        # insert all elements into the part.
        for elem in part.flat.getElementsByClass(('Note', 'Chord')):
            insert_elem_to_part(elem, temp_part)

        # make rests in places where there are no notes
        temp_part.makeRests(fillGaps=True, inPlace=True)

        m21_stream.insert(temp_part)


def insert_elem_to_part(elem: m21.chord.Chord, temp_part: VanillaPart):
    """
    some add-ons might be added here
    :param elem:
    :param temp_part:
    :return:
    """
    temp_part.insert_local(elem)


def pitch_set(elem: m21.note.GeneralNote) -> set:
    """
    calculates the pitch set for a note or a chord
    :param elem:
    :return:
    """
    return_set = set()
    for pitch in elem.pitches:
        return_set.add(pitch.ps)
    return return_set


def transpose_key(mxl_file: m21.stream.Score) -> bool:
    """
    transpose the key to C major or A minor by applying the Krumhansl-Schmuckler-algorithm
    already implemented in music21
    :param mxl_file:
    :return:
    """
    try:
        key = mxl_file.analyze('key')
    except m21.analysis.discrete.DiscreteAnalysisException:
        raise FileNotFittingSettingsError("INVALID_KEY")

    if key.mode != 'major' and key.mode != 'minor':
        return False

    if key.type == "major":
        interval_ = m21.interval.Interval(key.tonic, m21.pitch.Pitch('C'))
        mxl_file.transpose(interval_, inPlace=True)
    else:
        interval_ = m21.interval.Interval(key.tonic, m21.pitch.Pitch('A'))
        mxl_file.transpose(interval_, inPlace=True)
    return True


def check_valid_time(m21_stream: VanillaStream):
    """
    raises a FileNotFittingSettingsError if the time signature specified in settings
    is different from the one found in the stream
    :param m21_stream:
    :return:
    """
    valid_time = c.music_settings.valid_time

    try:
        if m21_stream.time_signature and m21_stream.time_signature.replace("/", "_") != valid_time:
            raise FileNotFittingSettingsError("WRONG_TIME_SIGNATURE")
    except AttributeError:
        # This happens if no time signature is specified - we then assume it to be 4/4,
        # which is the standard value
        pass


def check_valid_bpm(m21_stream: VanillaStream):
    """
    raises a FileNotFittingSettingsError if the beats per minute of this piece
    are not in the range specified in the settings file
    :param m21_stream:
    :return:
    """
    try:
        if (m21_stream.max_metronome > c.music_settings.max_bpm) or (
                m21_stream.min_metronome < c.music_settings.min_bpm):
            raise FileNotFittingSettingsError("WRONG_BPM")
    except TypeError:
        # this happens if there are no beats per minute specified. We then expect them to be 120,
        # as this is usually the standard value.
        pass


def process_data(thread_id, m21_stream):
    """
    make all the preprocessing from raw file to full Vanilla Stream
    :param thread_id:
    :param m21_stream:
    :return:
    """

    # print("%s processing %s" % (thread_id, m21_stream.id))
    m21_file = m21.converter.parse(m21_stream.id)

    make_file_container(m21_file=m21_file, m21_stream=m21_stream)

    check_valid_time(m21_stream)

    check_valid_bpm(m21_stream)

    process_file(m21_file, m21_stream)

    return m21_stream


def make_key_and_correlations(m21_stream: VanillaStream):
    """
    make all the preprocessing from a fully specified VanillaStream to a Stream where
    either a error is raised if the key doesn't fit the settings or
    all the insignificant parts are deleted and only the ones with the best
    correlation coefficient are kept
    :param m21_stream:
    :return:
    """
    if not transpose_key(m21_stream):
        raise FileNotFittingSettingsError("INVALID_KEY")

    try:
        stream_key = m21_stream.analyze('key')
    except m21.analysis.discrete.DiscreteAnalysisException:
        raise FileNotFittingSettingsError("INVALID_KEY")

    if stream_key.name != c.music_settings.accepted_key:
        raise FileNotFittingSettingsError("WRONG_KEY")

    for p in list(m21_stream.parts):
        try:
            part_key = p.analyze('key')
        except m21.analysis.discrete.DiscreteAnalysisException:
            part_key = deepcopy(stream_key)
            part_key.correlationCoefficient = -1.0

        if stream_key.name != part_key.name:
            for k in part_key.alternateInterpretations:
                if k.name == stream_key.name:
                    part_key = k
                    break

        if stream_key.name != part_key.name:
            part_key = deepcopy(stream_key)
            part_key.correlationCoefficient = -1.0

        if part_key.correlationCoefficient < c.music_settings.delete_part_threshold:
            m21_stream.remove(p, recurse=True)

    if len(m21_stream.parts) == 0:
        m21_stream.key = "invalid"
        m21_stream.key_correlation = -1.0
        raise FileNotFittingSettingsError("NO_PARTS")

    try:
        new_stream_key = m21_stream.analyze('key')
    except m21.analysis.discrete.DiscreteAnalysisException:
        raise FileNotFittingSettingsError("INVALID_KEY")

    if new_stream_key.name != stream_key.name or (new_stream_key.correlationCoefficient <
                                                  c.music_settings.delete_part_threshold):
        m21_stream.key = "invalid"
        m21_stream.key_correlation = -1.0
        raise FileNotFittingSettingsError("LOW_CORRELATION_KEY")

    m21_stream.key = new_stream_key.name
    m21_stream.key_correlation = new_stream_key.correlationCoefficient
    return
