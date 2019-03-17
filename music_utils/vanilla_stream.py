import music21 as m21


class VanillaStream(m21.stream.Score):
    """
    works like a m21.stream.Score, but has some extra methods that might be needed/are needed
    for analysis of musical pieces.
    """

    def __init__(self, filename=None):
        m21.stream.Stream.__init__(self)
        if filename:
            self.id = filename
        self.time_signature = None
        self.min_metronome = None
        self.max_metronome = None
        self.key = None
        self.key_correlation = None
        self.valid = False

    def insert_local(self, elem):
        if type(elem) == m21.meter.TimeSignature:
            self.insert_time_signature(elem)
        elif type(elem) == m21.tempo.MetronomeMark:
            self.insert_metronome_mark(elem)

    def insert_time_signature(self, elem: m21.meter.TimeSignature):
        super().insert(elem)
        temp_time = elem.ratioString
        if self.time_signature:
            if self.time_signature != "changing" and self.time_signature != temp_time:
                self.time_signature = "changing"
        else:
            self.time_signature = temp_time

    def insert_metronome_mark(self, elem: m21.tempo.MetronomeMark):
        super().insert(elem)
        if self.max_metronome is None:
            self.max_metronome = elem.number
            self.min_metronome = elem.number
        else:
            if elem.number > self.max_metronome:
                self.max_metronome = elem.number
            elif elem.number < self.max_metronome:
                self.min_metronome = elem.number
