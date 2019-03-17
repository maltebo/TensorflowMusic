def round_to_quarter(value):
    """
    rounds, for example offsets, lengths of notes
    :param value:
    :return:
    """
    return round(value * 4) / 4


class FileNotFittingSettingsError(BaseException):
    """
    utility class for error logging and information gain
    """
    def __init__(self, *args):
        super().__init__(*args)
