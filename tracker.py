import numpy as np

import time
from itertools import chain
from collections import defaultdict

from mapping import Map #, MappingThread
from optimization import BundleAdjustment
from measurements import MeasurementSource
from motion import MotionModel
from frame import StereoFrame, Frame
import g2o


class Tracker(object):
    def __init__(self, params, cam):
        self.params = params
        self.cam = cam

        self.motion_model = MotionModel()
        self.map = Map()
        # self.mapping_thread = MappingThread(self.map)
        # self.mapping_thread.start()
        
        self.preceding = None        # last keyframe
        self.current = None          # current frame
        self.status = defaultdict(bool)

        self.optimizer = BundleAdjustment()
        self.min_measurements = params.pnp_min_measurements
        self.max_iterations = params.pnp_max_iterations

    def initialize(self, frame):
        keyframe = frame.to_keyframe()
        mappoints, measurements = keyframe.create_mappoints_from_triangulation()

        assert len(mappoints) >= self.params.init_min_points, (
            'Not enough points to initialize map.')

        keyframe.set_fixed(True)
        self.map.add_keyframe(keyframe)

        for mappoint, measurement in zip(mappoints, measurements):
            self.map.add_mappoint(mappoint)
            self.map.add_point_measurement(keyframe, mappoint, measurement)
            keyframe.add_measurement(measurement)
            mappoint.add_measurement(measurement)
            mappoint.increase_measurement_count()

        maplines, line_measurements = keyframe.create_maplines_from_triangulation()
        print(f'Initialized {len(maplines)} lines')
        for mapline, measurement in zip(maplines, line_measurements):
            self.map.add_mapline(mapline)
            self.map.add_line_measurement(keyframe, mapline, measurement)
            keyframe.add_measurement(measurement)
            mapline.add_measurement(measurement)
            mapline.increase_measurement_count()

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

        origin = g2o.Isometry3d()
        left_frame = Frame(i, origin, self.cam, self.params, left_img, timestamp)
        right_frame = Frame(i, self.cam.compute_right_camera_pose(origin), self.cam, self.params, right_img, timestamp)
        frame = StereoFrame(left_frame, right_frame)

        if i == 0:
            self.initialize(frame)
            return

        self.current = frame

        predicted_pose, _ = self.motion_model.predict_pose(frame.timestamp)
        frame.update_pose(predicted_pose)

        local_mappoints = self.get_local_map_points(frame)
        measurements = frame.match_mappoints(local_mappoints)

        local_maplines = self.get_local_map_lines(frame)
        line_measurements = frame.match_maplines(local_maplines)

        tracked_map = set()
        for m in measurements:
            mappoint = m.get_map_primitive()
            mappoint.update_descriptor(m.get_descriptor())
            mappoint.increase_measurement_count()
            mappoint.add_measurement(m)
            tracked_map.add(mappoint)

        for m in line_measurements:
            mapline = m.get_map_primitive()
            mapline.update_descriptor(m.get_descriptor())
            mapline.add_measurement(m)
            mapline.increase_measurement_count()
        
        try:
            pose = self.refine_pose(frame.pose, self.cam, measurements)
            frame.update_pose(pose)
            self.motion_model.update_pose(frame.timestamp, pose.position(), pose.orientation())
            tracking_is_ok = True
        except:
            tracking_is_ok = False
            print('tracking failed!!!')

        if tracking_is_ok and self.should_be_keyframe(frame, measurements):
            self.create_new_keyframe(frame)

    def create_new_keyframe(self, frame):
            keyframe = frame.to_keyframe()
            keyframe.update_preceding(self.preceding)

            mappoints, measurements = keyframe.create_mappoints_from_triangulation()
            self.map.add_keyframe(keyframe)

            for mappoint, measurement in zip(mappoints, measurements):
                self.map.add_mappoint(mappoint)
                self.map.add_point_measurement(keyframe, mappoint, measurement)
                keyframe.add_measurement(measurement)
                mappoint.add_measurement(measurement)
                mappoint.increase_measurement_count()

            maplines, line_measurements = keyframe.create_maplines_from_triangulation()
            # frame.visualise_measurements(line_measurements)
            print(f'New Keyframe with {len(maplines)} lines')
            for mapline, measurement in zip(maplines, line_measurements):
                self.map.add_mapline(mapline)
                self.map.add_line_measurement(keyframe, mapline, measurement)
                keyframe.add_measurement(measurement)
                mapline.add_measurement(measurement)
                mapline.increase_measurement_count()
            
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

    def get_local_map_lines(self, frame):
        checked = set()
        filtered = []

        # Add in map points from preceding and reference
        for ln in self.preceding.maplines():  # neglect can_view test
            if ln in checked or ln.is_bad():
                continue
            ln.increase_projection_count()
            filtered.append(ln)

        return filtered


    def should_be_keyframe(self, frame, measurements):
        n_matches = len(measurements)
        n_matches_ref = len(self.preceding.measurements())

        return ((n_matches / n_matches_ref) < 
            self.params.min_tracked_points_ratio) or n_matches < 20