#!/usr/bin/env python3

import os
import random
import sys
import threading
import time
import traceback

import music_utils.simple_classes as simple
import preprocessing.melody_and_chords.find_melody as find_melody
import settings.constants as c
import settings.music_info_pb2 as music_info


class MakeDataThread(threading.Thread):
    """
    This Thread class is designed to make all the melody finding while communicating with other threads
    via a queue specified in constants, that holds all filenames ready to be processed (.pb files)
    """
    def __init__(self, thread_id=random.randint(100000, 999999)):
        threading.Thread.__init__(self)
        self.threadID = thread_id
        self.exit_flag = 0

    def run(self):
        self.make_melody_and_write_pb_file()

    def stop(self):
        self.exit_flag = 1

    def get_job(self) -> str:
        if not c.proto_buffer_work_queue.empty():
            file_name = c.proto_buffer_work_queue.get()
            return file_name

        else:
            time.sleep(1)
            return ""

    def make_melody_and_write_pb_file(self):

        while not c.proto_buffer_work_queue.empty() and not self.exit_flag:
            filename = self.get_job()

            if not filename:
                continue

            no_update = False

            try:

                new_filename = filename.replace('.pb', '_tf_skyline.melody_pb')

                if os.path.exists(new_filename):
                    no_update = True
                    continue

                with open(filename, 'rb') as fp:

                    proto_buffer = music_info.VanillaStreamPB()
                    proto_buffer.ParseFromString(fp.read())

                simple_song = simple.Song(proto_buffer=proto_buffer)

                melodies = find_melody.tf_skyline(simple_song, split=True)

                if melodies is None or len(melodies) == 0:
                    continue

                melody_list = music_info.MelodyList()
                melody_list.extra_info = "Doesn't save note volumes"
                melody_list.filepath = os.path.relpath(filename, c.MXL_DATA_FOLDER).replace('.pb', '.mxl')
                melody_list.algorithm = music_info.TF_SKYLINE

                for i, m in enumerate(melodies):
                    melody_part = melody_list.melodies.add()

                    melody_part.actual_start = m[0]

                    melody_part.offsets.extend([n.offset for n in m[1]])
                    melody_part.lengths.extend([n.length for n in m[1]])
                    melody_part.pitches.extend([n.pitch for n in m[1]])

                with open(new_filename, 'xb') as fp:

                    fp.write(melody_list.SerializeToString())

            except:
                print("\n\n\n", filename, "\n\n", sys.exc_info()[1], "\n\n", traceback.print_exc())

            finally:
                c.melody_lock.acquire()

                if no_update:
                    c.proto_buffers_to_do -= 1
                    if c.proto_buffers_done == 0:
                        c.proto_buffer_start_time = time.time()
                else:
                    c.proto_buffers_done += 1

                    current_time = time.time()

                    full_seconds_left = (((current_time - c.proto_buffer_start_time) / (c.proto_buffers_done + 0.001)) *
                                         (c.proto_buffers_to_do - c.proto_buffers_done))

                    days_left = str(round(full_seconds_left // 86400))

                    hours_left = str(round((full_seconds_left % 86400) // 3600))

                    minutes_left = str(round((full_seconds_left % 3600) // 60))

                    seconds_left = str(round(full_seconds_left % 60))

                    print("\rFinished {do:>5}/{todo}, {p:>7.3f} percent of the files, Time left: {d:>2}d, {h:>2}h, "
                          "{m:>2}min, {s:>2}s     ".format(
                        do=str(c.proto_buffers_done), todo=str(c.proto_buffers_to_do),
                        p=(round(c.proto_buffers_done / c.proto_buffers_to_do, 5)) * 100,
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

    c.proto_buffer_start_time = time.time()

    for t in threads:
        t.start()

    for t in threads:
        t.join()

    print("\n\n\nExiting all {n} Threads".format(n=thread_number))

    sys.exit(0)
