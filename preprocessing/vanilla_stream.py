from music21 import *


class VanillaStream(stream.Score):

    def __init__(self):
        stream.Stream.__init__(self)
        self.time_signature = None
        # we want to adjust the times so that one beat is approx. 0.5 seconds - 120bpm
        self.metronome_mark_min = None
        self.metronome_mark_max = None

    def insert_local(self, elem):
        if type(elem) == meter.TimeSignature:
            self.insert_time_signature(elem)
        elif type(elem) == tempo.MetronomeMark:
            self.insert_metronome_mark(elem)

    def insert_time_signature(self, elem: meter.TimeSignature):
        super().insert(elem)
        temp_time = elem.ratioString.replace("/", "_")
        if self.time_signature:
            if self.time_signature != "changing" and self.time_signature != temp_time:
                self.time_signature = "changing"
        else:
            self.time_signature = temp_time

    def insert_metronome_mark(self, elem: tempo.MetronomeMark):
        super().insert(elem)
        if self.metronome_mark_max is None:
            self.metronome_mark_max = elem.number
            self.metronome_mark_min = elem.number
        else:
            if elem.number > self.metronome_mark_max:
                self.metronome_mark_max = elem.number
            elif elem.number < self.metronome_mark_max:
                self.metronome_mark_min = elem.number
