#!/usr/bin/env python3
import os
import random
import sys
import threading
import time
import traceback

import music21 as m21

import settings.constants as c
from music_utils.vanilla_stream import VanillaStream
from preprocessing.analyze_and_modify.create_modified_stream import make_key_and_correlations
from preprocessing.analyze_and_modify.create_modified_stream import process_data
from preprocessing.analyze_and_modify.make_info import proto_buffer_entry_exists, save_vanilla_stream_pb
from preprocessing.analyze_and_modify.make_info import put_in_protocol_buffer, make_invalid_in_protocol_buffer
from preprocessing.helper import FileNotFittingSettingsError


class MakeDataThread(threading.Thread):
    """
    This Thread class is designed to make all the preprocessing while communicating with other threads
    via a queue specified in constants, that holds all filenames ready to be processed
    """

    def __init__(self, thread_id=random.randint(100000, 999999)):
        threading.Thread.__init__(self)
        self.threadID = thread_id
        self.work_queue = c.mxl_work_queue
        self.exit_flag = 0

    def run(self):
        self.run_analyze_and_create_data()

    def stop(self):
        self.exit_flag = 1

    def get_job(self) -> str:
        if not self.work_queue.empty():
            file_name = self.work_queue.get()
            return file_name

        else:
            time.sleep(1)
            return ""

    def run_analyze_and_create_data(self):
        """
        1. loads a file and creates a corresponding VanillaStream.
        2. saves the file information in the global protocol buffer
        3. saves a local pb file at the same place where the original
           file was located, with .pb ending
        :return:
        """
        while not self.work_queue.empty() and not self.exit_flag:
            filename = self.get_job()

            if not filename:
                continue

            no_update = False

            valid = False
            exists = False
            m21_stream = None

            try:

                exists, valid = proto_buffer_entry_exists(filename)
                if exists and not valid:
                    no_update = True
                    continue

                if exists and os.path.exists(filename.replace('.mxl', '.pb')):
                    no_update = True
                    continue

                m21_stream = VanillaStream(filename)

                process_data(self.threadID, m21_stream)
                make_key_and_correlations(m21_stream)
                m21_stream.valid = True

                piece_of_music_pb = None
                if c.UPDATE and not exists:
                    piece_of_music_pb = put_in_protocol_buffer(m21_stream)

                save_vanilla_stream_pb(m21_stream, piece_of_music_pb)

                # in the skyline algorithm it is asserted that the melody is a sequence
                # melody_stream = simple_skyline_algorithm(m21_stream)
                #
                # melody_struct = get_tf_structure(melody_stream)
                # save_melody_struct(filename, melody_struct)

                # melody_stream.show('midi')

                # full_stream = find_chords(m21_stream, melody_stream)
            except FileNotFittingSettingsError:
                if valid:
                    make_invalid_in_protocol_buffer(filename, str(sys.exc_info()[1]))
                    print(sys.exc_info()[1])
                    continue
                if c.UPDATE and not exists:
                    put_in_protocol_buffer(m21_stream, error_message=str(sys.exc_info()[1]))

            except m21.duration.DurationException:
                if valid:
                    make_invalid_in_protocol_buffer(filename, error='INVALID_FILE')
                    print(sys.exc_info()[1])
                    continue
                if c.UPDATE and not exists:
                    put_in_protocol_buffer(m21_stream, error_message='INVALID_FILE')

            except:
                print("\n\n\n", filename, "\n\n", sys.exc_info()[1], "\n\n", traceback.print_exc())

            finally:
                c.melody_lock.acquire()

                if no_update:
                    c.mxl_files_to_do -= 1
                    if c.mxl_files_done == 0:
                        c.mxl_start_time = time.time()
                else:
                    c.mxl_files_done += 1

                    current_time = time.time()

                    full_seconds_left = (((current_time - c.mxl_start_time) / (c.mxl_files_done + 0.001)) *
                                         (c.mxl_files_to_do - c.mxl_files_done))

                    days_left = str(round(full_seconds_left // 86400))

                    hours_left = str(round((full_seconds_left % 86400) // 3600))

                    minutes_left = str(round((full_seconds_left % 3600) // 60))

                    seconds_left = str(round(full_seconds_left % 60))

                    # print information about the probable time left until every file is processed
                    # highly annoying since its not actually decreasing a lot, as one can easily imagine :D
                    # I've seen 4d for a while now so I'm really thinking about deleting the feature again :P
                    print("\rFinished {do:>5}/{todo}, {p:>7.3f} percent of the files, Time left: {d:>2}d, {h:>2}h, "
                          "{m:>2}min, {s:>2}s     ".format(
                        do=str(c.mxl_files_done), todo=str(c.mxl_files_to_do),
                        p=(round(c.mxl_files_done / c.mxl_files_to_do, 5)) * 100,
                        d=days_left, h=hours_left, m=minutes_left, s=seconds_left),
                        file=sys.stdout, flush=True, end='')

                c.melody_lock.release()


if __name__ == "__main__":
    thread_number = 8
    threads = []

    for tName in range(thread_number):
        thread = MakeDataThread(tName + 1)
        threads.append(thread)

    print("Starting all {n} Threads\n\n\n".format(n=thread_number), flush=True)

    c.mxl_start_time = time.time()

    for t in threads:
        t.start()

    for t in threads:
        t.join()

    print("\n\n\nExiting all {n} Threads".format(n=thread_number))

    sys.exit(0)
