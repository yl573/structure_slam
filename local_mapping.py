import numpy as np

from queue import Queue
from threading import Thread, Lock, Condition
import time

from itertools import chain
from collections import defaultdict

from optimization import LocalBA
from components import Measurement

from multiprocessing import Process, Pipe
from enum import Enum


class DataType(Enum):
    KeyFrame = 1
    Measurement = 2

class PipeKeyFrame:
    def __init__(self, keyframe_id):
        self.id = keyframe_id
        self.pose = pose

class PipeMapPoint:
    def __init__(self, mappoint_id, position):
        self.id = mappoint_id
        self.position = position

class PipeMeasurement:
    def __init__(self, screen_pos, mappoint_id, keyframe_id):
        self.screen_pos = screen_pos
        self.mappoint_id = mappoint_id
        self.keyframe_id = keyframe_id


class MappingProcess():

    def __init__(self, cam):
        


def mapping_process(camera, conn: Pipe):

    optimizer = LocalBA()

    keyframes = []
    mappoints = []
    measurements = []

    stopped = False
    while not stopped:

        # we need two things, keyframe poses and measurements
        # the measurements have mappoints, keyframe id and projected position

        new_keyframes = []
        new_mappoints = []
        new_measurements = []
        while not conn.empty()
            new_data = conn.recv()
            if type(new_data) is PipeKeyFrame:
                new_keyframes.append(new_data)
            elif type(new_data) is PipeMapPoint:
                new_mappoints.append(new_data)
            elif type(new_data) is PipeMeasurement:
                new_measurements.append(new_data)

        for kf in new_keyframes:
            optimizer.add_keyframe(kf.id, kf.pose, camera, fixed=True)
        
        for mp in new_mappoints:
            optimizer.add_mappoint(mp.id, mp.position)
        

        

        