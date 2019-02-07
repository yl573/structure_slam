import numpy as np
from enum import Enum
from collections import defaultdict

class MapPoint:
    _id = 0

    def __init__(self, position, normal, descriptor,
                 color=np.zeros(3),
                 covariance=np.identity(3) * 1e-4):

        self.id = MapPoint._id
        MapPoint._id += 1

        self.position = position
        self.normal = normal
        self.descriptor = descriptor
        self.covariance = covariance
        self.color = color
        self.meas = list()

        self.count = defaultdict(int)

    def measurements(self):
        return list(self.meas)

    def add_measurement(self, m):
        self.meas.append(m)

    def update_position(self, position):
        self.position = position

    def update_normal(self, normal):
        self.normal = normal

    def update_descriptor(self, descriptor):
        self.descriptor = descriptor

    def set_color(self, color):
        self.color = color

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


class Measurement:

    Source = Enum('Measurement.Source', [
                  'TRIANGULATION', 'TRACKING', 'REFIND'])
    Type = Enum('Measurement.Type', ['STEREO', 'LEFT', 'RIGHT'])

    def __init__(self, type, source, mappoint, keypoints, descriptors):

        self.mappoint = mappoint

        self.type = type
        self.source = source
        self.keypoints = keypoints
        self.descriptors = descriptors
        self.view = None    # mappoint's position in current coordinates frame

        self.xy = np.array(self.keypoints[0])
        if self.type == self.Type.STEREO:
            self.xyx = np.array([
                *keypoints[0], keypoints[1][0]])

        self.triangulation = (source == self.Source.TRIANGULATION)

    @property
    def id(self):
        return (self.keyframe.id, self.mappoint.id)

    def __hash__(self):
        return hash(self.id)

    def get_descriptor(self, i=0):
        return self.descriptors[i]

    def get_keypoint(self, i=0):
        return self.keypoints[i]

    def get_descriptors(self):
        return self.descriptors

    def get_keypoints(self):
        return self.keypoints

    def is_stereo(self):
        return self.type == Measurement.Type.STEREO

    def is_left(self):
        return self.type == Measurement.Type.LEFT

    def is_right(self):
        return self.type == Measurement.Type.RIGHT

    def from_triangulation(self):
        return self.triangulation

    def from_tracking(self):
        return self.source == Measurement.Source.TRACKING

    def from_refind(self):
        return self.source == Measurement.Source.REFIND