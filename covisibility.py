from threading import Lock

from collections import defaultdict, Counter
from itertools import chain


class CovisibilityGraph(object):
    def __init__(self, ):
        self._lock = Lock()

        self.kfs = []
        self.pts = set()
        
        self.kfs_set = set()
        self.meas_lookup = dict()

    def keyframes(self):
        with self._lock:
            return self.kfs.copy()

    def mappoints(self):
        with self._lock:
            return self.pts.copy()

    def add_keyframe(self, kf):
        with self._lock:
            self.kfs.append(kf)
            self.kfs_set.add(kf)

    def add_mappoint(self, pt):
        with self._lock:
            self.pts.add(pt)

    def add_measurement(self, kf, pt, meas):
        with self._lock:
            if kf not in self.kfs_set or pt not in self.pts:
                return

            meas.keyframe = kf
            meas.mappoint = pt
            kf.add_measurement(meas)
            pt.add_measurement(meas)

            self.meas_lookup[meas.id] = meas

    def has_measurement(self, *args):
        with self._lock:
            if len(args) == 1:                                 # measurement
                return args[0].id in self.meas_lookup
            elif len(args) == 2:                               # keyframe, mappoint
                id = (args[0].id, args[1].id)
                return id in self.meas_lookup
            else:
                raise TypeError
