from statistics import mean

import music21 as m21

from preprocessing.helper import round_to_quarter


class VanillaPart(m21.stream.Part):
    """
    works like a m21.stream.Part, but has some extra information and methods that might be needed/are needed
    for analysis of musical pieces.
    """

    def __init__(self):
        m21.stream.Part.__init__(self)
        self._pitch_list = []
        self._volume_list = []
        self._note_number = 0
        self._total_notes_or_chords = 0
        self._total_pitches = 0
        self._lyrics_number = 0
        self._note_lengths = []
        self._key = None
        self._key_correlation = None
        self._changed = False
        self._note_percentage = None
        self._lyrics_percentage = None
        self._average_pitch = None
        self._average_volume = None

    def insert_local(self, elem, new_duration=None):
        self._changed = True
        if type(elem) == m21.note.Note:
            self.insert_note(elem, new_duration)
        elif type(elem) == m21.chord.Chord or type(elem) == m21.harmony.ChordSymbol:
            self.insert_chord(elem, new_duration)
        else:
            print(type(elem))
            raise ValueError("That was not the thing we expected you to do...")

    def insert_note(self, elem: m21.note.Note, new_duration: float):
        start = elem.offset
        end = start + elem.quarterLength
        if new_duration:
            end = start + new_duration
        temp_volume = elem.volume.velocity
        temp_lyrics = elem.lyrics
        temp_pitch = elem.pitch.ps

        temp_note = self.create_note(start, end, temp_volume, temp_lyrics, temp_pitch)

        if not temp_note:
            return

        self._pitch_list.append(temp_pitch)
        self._volume_list.append(temp_volume)
        self._note_number += 1
        self._total_notes_or_chords += 1
        self._total_pitches += 1
        try:
            self._note_lengths.append(elem.seconds)
        except m21.exceptions21.Music21Exception:
            pass

        if temp_lyrics:
            self._lyrics_number += 1

        super().insert(temp_note)

    def insert_chord(self, elem: m21.chord.Chord, new_duration: float):
        start = elem.offset
        end = start + elem.quarterLength
        if new_duration:
            end = start + new_duration
        temp_volume = elem.volume.velocity
        temp_lyrics = elem.lyrics

        for temp_full_pitch in elem.pitches:

            temp_pitch = temp_full_pitch.ps

            temp_note = self.create_note(start, end, temp_volume, temp_lyrics, temp_pitch)

            if not temp_note:
                return

            self._pitch_list.append(temp_pitch)
            self._total_pitches += 1
            super().insert(temp_note)

        self._volume_list.append(temp_volume)
        self._total_notes_or_chords += 1

        if temp_lyrics:
            self._lyrics_number += 1

    @property
    def key(self):
        if (not self._key) or self._changed:
            self.calculate_attributes()

        return self._key.name

    def key_by_name(self, key_name: str) -> m21.key.Key:

        if self.key == key_name:
            return self._key

        for k in self._key.alternateInterpretations:
            if k.name == key_name:
                return k

        raise ValueError("No key matching your key name {n} was found".format(n=key_name))

    @property
    def key_correlation(self):
        if (not self._key) or self._changed:
            self.calculate_attributes()

        return self._key.correlationCoefficient

    @property
    def lyrics_percentage(self):
        if (not self._lyrics_percentage) or self._changed:
            self.calculate_attributes()

        return self._lyrics_percentage

    @lyrics_percentage.setter
    def lyrics_percentage(self, value):
        assert 0.0 <= value <= 1.0
        self._lyrics_percentage = value

    @property
    def note_percentage(self):
        if (not self._note_percentage) or self._changed:
            self.calculate_attributes()

        return self._note_percentage

    @note_percentage.setter
    def note_percentage(self, value):
        assert 0.0 <= value <= 1.0
        self._note_percentage = value

    @property
    def average_pitch(self):
        if (not self._average_pitch) or self._changed:
            self.calculate_attributes()

        return self._average_pitch

    @property
    def average_volume(self):
        if (not self._average_volume) or self._changed:
            self.calculate_attributes()

        return self._average_volume

    def calculate_attributes(self):
        if self._total_notes_or_chords:
            self._note_percentage = self._note_number / self._total_notes_or_chords
            self._lyrics_percentage = self._lyrics_number / self._total_notes_or_chords
        else:
            self._note_percentage = 0.0
            self._lyrics_percentage = 0.0

        self._average_pitch = mean(self._pitch_list)
        self._average_volume = mean(self._volume_list)
        self._key = self.analyze('key')

        self._changed = False

    @staticmethod
    def create_note(start, end, temp_volume: int, temp_lyrics, temp_pitch):
        if end - start < 0.2:
            return None

        if 0 > temp_pitch > 128:
            return None

        # maximum note length is 4
        if end - start > 4:
            end = start + 4

        temp_note = m21.note.Note()

        temp_note.offset = round_to_quarter(start)
        new_end = round_to_quarter(end)
        new_duration = new_end - temp_note.offset
        if new_duration < 0.25:
            return None

        temp_note.quarterLength = new_duration

        temp_note.volume.velocity = temp_volume
        if temp_lyrics:
            temp_note.lyrics = temp_lyrics

        temp_note.pitch.ps = temp_pitch

        return temp_note
