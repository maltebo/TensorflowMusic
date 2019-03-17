#!/usr/bin/env python3

import os
import queue
import threading
import time

import settings.music_info_pb2 as music_info
from settings.music_info_pb2 import Settings

print("start variable setup")
start_time = time.time()

home_directory = "/home/malte/PycharmProjects/TensorflowMusic"

os.chdir(home_directory)

DATA_FOLDER = os.path.join(home_directory, "data")
MXL_FOLDER = os.path.join(home_directory, "data/MXL")
MXL_DATA_FOLDER = os.path.join(home_directory, "data/MXL/lmd_matched_mxl")

MUSIC_INFO_FOLDER = os.path.join(home_directory, "data/music_info_pb")

for folder in [DATA_FOLDER, MXL_FOLDER, MXL_DATA_FOLDER, MUSIC_INFO_FOLDER]:
    try:
        os.mkdir(folder)
    except FileExistsError:
        pass

UPDATE = True
UPDATE_FREQUENCY = 10

DRAFT = False


def make_settings() -> Settings:
    settings = Settings()
    settings.min_pitch = 49.0
    settings.max_pitch = 84.0
    settings.delete_part_threshold = 0.65
    settings.delete_stream_threshold = 0.8
    settings.accepted_key = "C major"
    settings.min_bpm = 100
    settings.max_bpm = 140
    settings.valid_time = "4_4"
    return settings


music_settings = make_settings()

if DRAFT:
    settings_filename = 'DRAFT_'
else:
    settings_filename = ''
settings_filename += str(round(music_settings.delete_part_threshold, 2))
settings_filename += '_' + str(round(music_settings.delete_stream_threshold, 2))
settings_filename += '_' + music_settings.accepted_key
settings_filename += '_' + str(music_settings.min_bpm)
settings_filename += '_' + str(music_settings.max_bpm)
settings_filename += '_' + music_settings.valid_time

settings_filename += '.pb'

music_info_dict_lock = threading.Lock()
music_info_file_lock = threading.Lock()
melody_lock = threading.Lock()

music_protocol_buffer = None

for dirpath, _, filenames in os.walk(MUSIC_INFO_FOLDER):

    for filename in filenames:

        if filename == settings_filename:
            if os.path.getsize(os.path.join(dirpath, filename)) == 0:
                os.remove(os.path.join(dirpath, filename))
                break
            with open(os.path.join(dirpath, filename), 'rb') as fp:
                music_protocol_buffer = music_info.MusicList()
                music_protocol_buffer.ParseFromString(fp.read())
                music_protocol_buffer.counter = 0
            break

existing_files = {}

PROTOCOL_BUFFER_LOCATION = os.path.join(MUSIC_INFO_FOLDER, settings_filename)

mxl_work_queue = queue.Queue(0)
mxl_files_done = 0

proto_buffer_work_queue = queue.Queue(0)
proto_buffers_done = 0

melody_work_queue = queue.Queue(0)
melodies_done = 0

for root, dirs, files in os.walk(MXL_DATA_FOLDER):
    for file in files:
        if file.endswith('.mxl'):
            mxl_work_queue.put(os.path.join(root, file))
        elif file.endswith('.pb'):
            proto_buffer_work_queue.put(os.path.join(root, file))
        elif file.endswith(('.melody_pb')):
            melody_work_queue.put(os.path.join(root, file))
            # take for every song only one version into account!
            break

mxl_files_to_do = mxl_work_queue.qsize()
mxl_start_time = time.time()

proto_buffers_to_do = proto_buffer_work_queue.qsize()
proto_buffer_start_time = mxl_start_time

melodies_to_do = melody_work_queue.qsize()
melodies_start_time = mxl_start_time


def make_music_list():
    ml = music_info.MusicList(settings=music_settings, counter=0)
    return ml


if not music_protocol_buffer:

    print("create music info file")

    music_protocol_buffer = make_music_list()

    with open(PROTOCOL_BUFFER_LOCATION, 'xb') as fp:
        fp.write(music_protocol_buffer.SerializeToString())

else:
    for f in music_protocol_buffer.music_data:
        existing_files[f.filepath] = f.valid

##################################################################
################ MODEL CONSTANTS #################################
##################################################################

sequence_length = 30


print("finished setup in {sec} seconds".format(sec=str(round(time.time() - start_time, 2))))
