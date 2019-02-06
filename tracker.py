import numpy as np

import time
from itertools import chain
from collections import defaultdict

from mapping import Map
from optimization import BundleAdjustment
from frame import Measurement
from motion import MotionModel
from frame import StereoFrame
import g2o


class Tracker(object):
    def __init__(self, params, cam):
        self.params = params
        self.cam = cam

        self.motion_model = MotionModel()
        self.map = Map()
        
        self.preceding = None        # last keyframe
        self.current = None          # current frame
        self.status = defaultdict(bool)

        self.optimizer = BundleAdjustment()
        self.min_measurements = params.pnp_min_measurements
        self.max_iterations = params.pnp_max_iterations

    def initialize(self, frame):
        mappoints, measurements = frame.create_mappoints_from_triangulation()
        assert len(mappoints) >= self.params.init_min_points, (
            'Not enough points to initialize map.')

        keyframe = frame.to_keyframe()
        keyframe.set_fixed(True)
        self.map.add_keyframe(keyframe)
        for mappoint, measurement in zip(mappoints, measurements):
            self.map.add_mappoint(mappoint)
            self.map.add_measurement(keyframe, mappoint, measurement)
            mappoint.increase_measurement_count()

        self.preceding = keyframe
        self.current = keyframe
        self.status['initialized'] = True

        self.motion_model.update_pose(
            frame.timestamp, frame.position, frame.orientation)

    def refine_pose(self, pose, cam, measurements):
        assert len(measurements) >= self.min_measurements, (
            'Not enough points')
            
        self.optimizer.clear()
        self.optimizer.add_pose(0, pose, cam, fixed=False)

        for i, m in enumerate(measurements):
            self.optimizer.add_point(i, m.mappoint.position, fixed=True)
            self.optimizer.add_edge(0, i, 0, m)

        self.optimizer.optimize(self.max_iterations)
        return self.optimizer.get_pose(0)

    
    def update(self, i, left_img, right_img, timestamp):

        frame = StereoFrame(i, g2o.Isometry3d(), self.cam, self.params, left_img, right_img, timestamp=timestamp)

        if i == 0:
            self.initialize(frame)
            return

        self.current = frame

        predicted_pose, _ = self.motion_model.predict_pose(frame.timestamp)
        
        frame.update_pose(predicted_pose)

        local_mappoints = self.get_local_map_points(frame)
        measurements = frame.match_mappoints(local_mappoints, Measurement.Source.TRACKING)

        tracked_map = set()
        for m in measurements:
            mappoint = m.mappoint
            mappoint.update_descriptor(m.get_descriptor())
            mappoint.increase_measurement_count()
            tracked_map.add(mappoint)
        
        try:
            pose = self.refine_pose(frame.pose, self.cam, measurements)
            frame.update_pose(pose)
            self.motion_model.update_pose(frame.timestamp, pose.position(), pose.orientation())
            tracking_is_ok = True
        except:
            tracking_is_ok = False
            print('tracking failed!!!')

        if tracking_is_ok and self.should_be_keyframe(frame, measurements):
            keyframe = frame.to_keyframe()
            keyframe.update_preceding(self.preceding)

            mappoints, measurements = keyframe.create_mappoints_from_triangulation()
            self.map.add_keyframe(keyframe)

            for mappoint, measurement in zip(mappoints, measurements):
                self.map.add_mappoint(mappoint)
                self.map.add_measurement(keyframe, mappoint, measurement)
                mappoint.increase_measurement_count()
            
            self.preceding = keyframe

    def get_local_map_points(self, frame):
        checked = set()
        filtered = []

        # Add in map points from preceding and reference
        for pt in self.preceding.mappoints():  # neglect can_view test
            if pt in checked or pt.is_bad():
                continue
            pt.increase_projection_count()
            filtered.append(pt)

        return filtered


    def should_be_keyframe(self, frame, measurements):
        n_matches = len(measurements)
        n_matches_ref = len(self.preceding.measurements())

        return ((n_matches / n_matches_ref) < 
            self.params.min_tracked_points_ratio) or n_matches < 20