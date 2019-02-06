import numpy as np

from queue import Queue
from threading import Thread, Lock, Condition
import time

from itertools import chain
from collections import defaultdict

from optimization import LocalBA
from frame import Measurement



class Mapping(object):
    def __init__(self, graph, params):
        self.graph = graph
        self.params = params

    def add_keyframe(self, keyframe, measurements):
        self.graph.add_keyframe(keyframe)
        self.create_points(keyframe)

        for m in measurements:
            self.graph.add_measurement(keyframe, m.mappoint, m)

    def create_points(self, keyframe):
        mappoints, measurements = keyframe.triangulate()
        self.add_measurements(keyframe, mappoints, measurements)

    def add_measurements(self, keyframe, mappoints, measurements):
        for mappoint, measurement in zip(mappoints, measurements):
            self.graph.add_mappoint(mappoint)
            self.graph.add_measurement(keyframe, mappoint, measurement)
            mappoint.increase_measurement_count()