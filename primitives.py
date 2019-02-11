import numpy as np
from collections import defaultdict
from random import randrange

class MapLandmarkBase:
    _id = 0

    def __init__(self):
        self.meas = list()
        self.count = defaultdict(int)

        self.id = MapLandmarkBase._id
        MapLandmarkBase._id += 1

    def measurements(self):
        return list(self.meas)

    def add_measurement(self, m):
        self.meas.append(m)

    def is_bad(self):
        status = (
            self.count['meas'] == 0
            or (self.count['outlier'] > 20
                and self.count['outlier'] > self.count['inlier'])
            or (self.count['proj'] > 20
                and self.count['proj'] > self.count['meas'] * 10))
        return status

    def increase_outlier_count(self):
        self.count['outlier'] += 1

    def increase_inlier_count(self):
        self.count['inlier'] += 1

    def increase_projection_count(self):
        self.count['proj'] += 1

    def increase_measurement_count(self):
        self.count['meas'] += 1

    def is_point(self):
        return type(self) == MapPoint
    
    def is_line(self):
        return type(self) == MapLine


class MapPoint(MapLandmarkBase):

    def __init__(self, position, normal, descriptor, color):
        super().__init__()

        self.position = position
        self.normal = normal
        self.descriptor = descriptor
        self.color = color
    
    def update_position(self, position):
        self.position = position

    def update_normal(self, normal):
        self.normal = normal

    def update_descriptor(self, descriptor):
        self.descriptor = descriptor

    def set_color(self, color):
        self.color = color


class MapLine(MapLandmarkBase):

    def __init__(self, endpoints, descriptor):
        super().__init__()

        self.endpoints = endpoints
        self.descriptor = descriptor

        self.color = (randrange(255), randrange(255), randrange(255))

    def update_endpoints(self, endpoints):
        self.endpoints = endpoints

    def update_descriptor(self, descriptor):
        self.descriptor = descriptor
