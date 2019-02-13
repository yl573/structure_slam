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

        self.keyframe_id = None

    @property
    def id(self):
        return (self.keyframe_id, self.map_primitive.id)

    def data(self):
        return NotImplementedError()

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

    def is_point(self):
        return type(self) is PointMeasurement

    def is_line(self):
        return type(self) is LineMeasurement


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

    def data(self):
        if self.is_stereo():
            return self.xyx
        return self.xy

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

    def __init__(self, measurement_type, source, mapline, keylines, descriptors):

        super().__init__(measurement_type, source, mapline)

        self.keylines = keylines
        self.descriptors = descriptors
        self.view = None    # mappoint's position in current coordinates frame

    @property
    def mapline(self):
        return self.map_primitive

    def data(self):
        return self.keylines

    @mapline.setter
    def mapline(self, value):
        assert type(value) is MapLine
        self.map_primitive = value

    def get_descriptor(self, i=0):
        return self.descriptors[i]