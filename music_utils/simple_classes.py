import os
from copy import deepcopy
from operator import attrgetter

import music21 as m21
from numpy import mean

import settings.constants as c
import settings.music_info_pb2 as music_info


class NoteList(list):
    """
    a list extension with the upgrades of a predefined sort function and
    the possibility to create a m21 Stream from it
    """

    def __init__(self, seq=()):
        super().__init__(seq)
        self._m21_stream = None
        self.id = float('inf')

    def sort(self, **kwargs):
        super().sort(key=attrgetter('pitch'), reverse=True)
        super().sort(key=attrgetter('length'), reverse=True)
        return super().sort(key=attrgetter('offset'))

    @property
    def m21_stream(self):
        if not self._m21_stream:
            self._m21_stream = m21.stream.Stream()
            for note in self:
                self._m21_stream.insert(note.m21_note)
        return self._m21_stream


class Part:
    """
    a simple part representation which can be initialised with various values (see type hints)
    smaller and faster than the m21 Implementation
    """
    def __init__(self, id: int, proto_buffer: music_info.VanillaPartPB = None,
                 info: music_info.PieceOfMusic.Part = None,
                 note_list: NoteList = None, name: str = ""):

        self._notes = NoteList()

        self.id = id

        if proto_buffer is not None and info is not None:
            self.name = proto_buffer.name

            self.average_pitch = info.average_pitch

            self.average_volume = info.average_volume
            self.key_correlation = info.key_correlation

            self.note_percentage = info.note_percentage
            self.lyrics_percentage = info.lyrics_percentage

            for offset, length, pitch, volume in zip(proto_buffer.offsets,
                                                     proto_buffer.lengths,
                                                     proto_buffer.pitches,
                                                     proto_buffer.volumes):
                self._notes.append(Note(offset, length, pitch, volume, self.id))

        elif note_list is not None:
            self.name = name
            weighted_notes = [n for n in note_list if 0 <= n.pitch <= 128 for _ in range(int(n.length * 4))]
            self.average_pitch = mean([n.pitch for n in weighted_notes])
            self.average_volume = mean([n.volume for n in weighted_notes])

            # Todo: Krumhansl-Algorithm?
            self.key_correlation = None
            self.note_percentage = None
            self.lyrics_percentage = None

            self._notes = deepcopy(note_list)


        else:
            raise ValueError("Please only call this with either proto buffer AND info OR"
                             "notelists and optionally a name")

        self._m21_part = None

        self._notes.sort()

    def notes(self, exclude_rests=False):
        if exclude_rests:
            return [note for note in self._notes if 0 <= note[2] <= 128]
        return deepcopy(self._notes)

    @property
    def m21_part(self):
        if not self._m21_part:
            self._m21_part = m21.stream.Part()
            self._m21_part.partName = self.name
            instrument = m21.instrument.Instrument()
            instrument.partName = self.name
            instrument.instrumentName = self.name
            self._m21_part.insert(instrument)
            for note in self._notes:
                self._m21_part.insert(note.m21_note)
        return self._m21_part

    def __str__(self):

        str_result = "\tName: {name}\n\t\tAverage Pitch: {avg_p}\n\t\tAverage Volume: {avg_v}" \
                     "\n\t\tKey-Correlation: {key_corr}\n\t\tNumber of notes: {note_nr}\n\t\t" \
                     "Number of rests: {rest_nr}\n".format(name=self.name, avg_p=self.average_pitch,
                                                           avg_v=self.average_volume, key_corr=self.key_correlation,
                                                           note_nr=len(
                                                               [1 for note in self._notes if 0 <= note[2] <= 128]),
                                                           rest_nr=len([0 for note in self._notes if note[2] == 200]))
        return str_result


class Note:
    # works like a list, and has 5 entries that can be called by properties
    def __init__(self, offset, length, pitch: 200, volume: 0, part: 2 ** 32):
        self._values = [offset, length, pitch, volume, part]
        self._m21_note = None

    @property
    def offset(self):
        return self._values[0]

    @offset.setter
    def offset(self, value):
        assert value >= 0.0
        self._values[0] = value

    @property
    def length(self):
        return self._values[1]

    @length.setter
    def length(self, value):
        assert value > 0.0
        self._values[1] = value

    @property
    def pitch(self):
        return self._values[2]

    @pitch.setter
    def pitch(self, value):
        assert value >= 0
        self._values[2] = value

    @property
    def volume(self):
        return self._values[3]

    @volume.setter
    def volume(self, value):
        assert value >= 0
        self._values[3] = value

    @property
    def part(self):
        return self._values[4]

    @part.setter
    def part(self, value):
        self._values[4] = value

    def end(self):
        return self.offset + self.length

    @property
    def m21_note(self):
        if not self._m21_note:
            if self.pitch == 200:
                self._m21_note = m21.note.Rest()
            else:
                self._m21_note = m21.note.Note()
                self._m21_note.pitch.ps = self.pitch
                self._m21_note.volume.velocity = self.volume
            self._m21_note.offset = self.offset
            self._m21_note.quarterLength = self.length
        return self._m21_note


class Song:
    """
    a simple Song representation which can be initialised with various values (see type hints)
    smaller and faster than the m21 (Stream) Implementation
    """
    def __init__(self, proto_buffer: music_info.VanillaStreamPB = None, list_of_parts_or_note_lists: list = None,
                 name: str = ""):

        self.parts = []
        self._notes = NoteList()

        self._m21_stream = None

        if proto_buffer is not None:
            self.name = proto_buffer.filepath

            self.key = proto_buffer.info.key
            self.key_correlation = proto_buffer.info.key_correlation

            for i, part in enumerate(proto_buffer.parts):

                found = False

                for p in proto_buffer.info.parts:

                    if part.name == p.name:
                        new_part = Part(id=i, proto_buffer=part, info=p)
                        self.parts.append(new_part)
                        self._notes.extend(new_part.notes(exclude_rests=True))
                        found = True
                        break

                if not found:
                    for p in proto_buffer.info.parts:

                        if part.name.startswith(p.name):
                            new_part = Part(id=i, proto_buffer=part, info=p)
                            self.parts.append(new_part)
                            self._notes.extend(new_part.notes(exclude_rests=True))
                            found = True
                            break

                assert found, "Part {p} wasn't found in info file\n{info}".format(p=part.name,
                                                                                  info=proto_buffer.info)

        elif list_of_parts_or_note_lists is not None:

            self.name = name

            for elem in list_of_parts_or_note_lists:
                if type(elem) == Part:
                    self.parts.append(deepcopy(elem))
                    self._notes.extend(elem.notes)
                elif type(elem) == NoteList:
                    id = 2 ** 32
                    if elem:
                        id = elem[0].part
                    self.parts.append(Part(id=id, note_list=elem))
                    self._notes.extend(elem)
                else:
                    raise ValueError("Type must be Part or NoteList, your type was {t}".format(t=type(elem)))

        self._notes.sort()

    def notes(self, exclude_rests=False):
        if exclude_rests:
            return [note for note in self._notes if 0 <= note[2] <= 128]
        return deepcopy(self._notes)

    def m21_stream(self):
        if not self._m21_stream:
            self._m21_stream = m21.stream.Stream()
            self._m21_stream.id = self.name
            for part in self.parts:
                self._m21_stream.insert(part.m21_part)
        return self._m21_stream

    def __str__(self):
        str_result = "Name: {name}\nKey Correlation: {key_corr}\nParts:\n".format(name=self.name,
                                                                                  key_corr=self.key_correlation)
        for part in self.parts:
            str_result += str(part)

        str_result += "\n"

        return str_result


if __name__ == '__main__':
    proto_buffer_path = os.path.join(c.MXL_DATA_FOLDER, "L/M/Y/TRLMYOF128F931D661/34b1b5457161a6a2509b386147e16177.pb")

    with open(proto_buffer_path, 'rb') as fp:
        proto_buffer = music_info.VanillaStreamPB()
        proto_buffer.ParseFromString(fp.read())
        assert proto_buffer.HasField('info')

    simple_song = Song(proto_buffer)

    print(type(simple_song))
