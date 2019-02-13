class Map(object):
    def __init__(self, ):
        self.kfs = []
        self.pts = set()
        self.lns = set()
        
        self.kfs_set = set()
        self.meas_lookup = dict()

    def keyframes(self):
        return self.kfs.copy()

    def mappoints(self):
        return self.pts.copy()

    def add_keyframe(self, kf):
        self.kfs.append(kf)
        self.kfs_set.add(kf)

    def add_mappoint(self, pt):
        self.pts.add(pt)

    def add_mapline(self, ln):
        self.lns.add(ln)

    def add_point_measurement(self, kf, pt, meas):
        if kf not in self.kfs_set or pt not in self.pts:
            return

        meas.keyframe = kf
        meas.mappoint = pt

        self.meas_lookup[meas.id] = meas

    def add_line_measurement(self, kf, ln, meas):
        if kf not in self.kfs_set or ln not in self.lns:
            return
        meas.keyframe = kf
        meas.mapline = ln

        self.meas_lookup[meas.id] = meas
