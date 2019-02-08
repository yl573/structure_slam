import numpy as np
from enum import Enum
from primitives import MapLandmarkBase, MapPoint, MapLine


class MeasurementSource(Enum):
    TRIANGULATION = 1
    TRACKING = 2
    REFIND = 3


class MeasurementType(Enum):
    STEREO = 1
    LEFT = 2
    RIGHT = 3


class MeasurementBase:

    def __init__(self, measurement_type, source, map_primitive : MapLandmarkBase):

        self.type = measurement_type
        self.source = source
        self.map_primitive = map_primitive

        self.keyframe = None

    @property
    def id(self):
        return (self.keyframe.id, self.map_primitive.id)

    def __hash__(self):
        return hash(self.id)

    def get_map_primitive(self):
        return self.map_primitive

    def is_stereo(self):
        return self.type == MeasurementType.STEREO

    def is_left(self):
        return self.type == MeasurementType.LEFT

    def is_right(self):
        return self.type == MeasurementType.RIGHT

    def from_triangulation(self):
        return self.source == MeasurementSource.TRIANGULATION

    def from_tracking(self):
        return self.source == MeasurementSource.TRACKING

    def from_refind(self):
        return self.source == MeasurementSource.REFIND


class PointMeasurement(MeasurementBase):

    def __init__(self, measurement_type, source, mappoint, keypoints, descriptors):

        super().__init__(measurement_type, source, mappoint)

        self.keypoints = keypoints
        self.descriptors = descriptors
        self.view = None    # mappoint's position in current coordinates frame

        self.xy = np.array(self.keypoints[0])
        if self.is_stereo():
            self.xyx = np.array([
                *keypoints[0], keypoints[1][0]])

    @property
    def mappoint(self):
        return self.map_primitive

    @mappoint.setter
    def mappoint(self, value):
        assert type(value) is MapPoint
        self.map_primitive = value

    def get_descriptor(self, i=0):
        return self.descriptors[i]


class LineMeasurement(MeasurementBase):

    def __init__(self, measurement_type, source, mapline, frame, keylines, descriptors):

        super().__init__(measurement_type, source, frame, mapline)

        self.keylines = keylines
        self.descriptors = descriptors
        self.view = None    # mappoint's position in current coordinates frame

    @property
    def mapline(self):
        return self.map_primitive

    @mapline.setter
    def mapline(self, value):
        assert type(value) is MapLine
        self.map_primitive = value