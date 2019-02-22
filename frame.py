import numpy as np
import cv2
import g2o

from threading import Lock, Thread
from queue import Queue

from enum import Enum
from collections import defaultdict
from numbers import Number
from primitives import MapPoint, MapLine
from measurements import PointMeasurement, LineMeasurement, MeasurementType, MeasurementSource
from line_algorithms import triangulate_lines
import math
from segmentation import SegmentationModel


segmentation = SegmentationModel()


class FeatureType(Enum):
    Point = 0
    Line = 1


class Frame(object):
    def __init__(self, idx, pose, cam, params, image, timestamp=None):
        self.idx = idx
        self.pose = pose    # g2o.Isometry3d
        self.cam = cam
        self.timestamp = timestamp
        self.params = params

        self.detector = params.feature_detector
        self.extractor = params.descriptor_extractor
        self.matcher = params.descriptor_matcher

        self.cell_size = params.matching_cell_size
        self.distance = params.matching_distance
        self.neighborhood = (
            params.matching_cell_size * params.matching_neighborhood)
        
        self.transform_matrix = pose.inverse().matrix()[:3]  # shape: (3, 4)
        self.projection_matrix = (
            self.cam.intrinsic.dot(self.transform_matrix))  # from world frame to image

        self.image = image
        self.height, self.width = image.shape[:2]
        self.keypoints, self.descriptors = self.extract_features(image)
        self.colors = self.get_color(self.keypoints, image)
        self.unmatched_points = np.ones(len(self.keypoints), dtype=bool)

        self.line_distance = params.line_matching_distance
        self.line_detector = params.line_detector
        self.line_extractor = params.line_extractor
        self.keylines, self.line_descriptors = self.extract_lines(image)
        self.unmatched_lines = np.ones(len(self.keylines), dtype=bool)

    def extract_features(self, image):
        keypoints = self.detector.detect(image)
        keypoints, descriptors = self.extractor.compute(image, keypoints)  
        keypoints = np.array([kp.pt for kp in keypoints])
        return keypoints, descriptors 

    def extract_lines(self, image):
        keylines = self.line_detector.detect(image)
        keylines, line_descriptors = self.line_extractor.compute(image, keylines)  
        keylines = np.array([[kl.startPointX, kl.startPointY, kl.endPointX, kl.endPointY] for kl in keylines])
        return keylines, line_descriptors  

    def can_view(self, points, ground=False, margin=20):    # Frustum Culling
        points = np.transpose(points)
        (u, v), depth = self.project(self.transform(points))

        if ground:
            return np.logical_and.reduce([
                depth >= self.cam.frustum_near,
                depth <= self.cam.frustum_far,
                u >= - margin,
                u <= self.cam.width + margin])
        else:
            return np.logical_and.reduce([
                depth >= self.cam.frustum_near,
                depth <= self.cam.frustum_far,
                u >= - margin,
                u <= self.cam.width + margin,
                v >= - margin,
                v <= self.cam.height + margin])

    @property
    def orientation(self):
        return self.pose.orientation()

    @property
    def position(self):
        return self.pose.position()

    def update_pose(self, pose):
        if isinstance(pose, g2o.SE3Quat):
            self.pose = g2o.Isometry3d(pose.orientation(), pose.position())
        else:
            self.pose = pose

        self.transform_matrix = self.pose.inverse().matrix()[:3]
        self.projection_matrix = (
            self.cam.intrinsic.dot(self.transform_matrix))

    def transform(self, points):    # from world coordinates
        '''
        Transform points from world coordinates frame to camera frame.
        Args:
            points: a point or an array of points, of shape (3,) or (3, N).
        '''
        R = self.transform_matrix[:3, :3]
        if points.ndim == 1:
            t = self.transform_matrix[:3, 3]
        else:
            t = self.transform_matrix[:3, 3:]
        return R.dot(points) + t

    def project(self, points):
        '''
        Project points from camera frame to image's pixel coordinates.
        Args:
            points: a point or an array of points, of shape (3,) or (3, N).
        Returns:
            Projected pixel coordinates, and respective depth.
        '''
        projection = self.cam.intrinsic.dot(points / points[-1:])
        return projection[:2], points[-1]

    def find_matches(self, points, descriptors):
        '''
        Match to points from world frame.
        Args:
            points: a list/array of points. shape: (N, 3)
            descriptors: a list of feature descriptors. length: N
        Returns:
            List of successfully matched (queryIdx, trainIdx) pairs.
        '''
        points = np.transpose(points)
        projection, _ = self.project(self.transform(points))
        projection = projection.transpose()

        matches = self.matcher.match(np.array(descriptors), self.descriptors)

        # distances are there to cope with multiple trainIdx matches
        good_matches = []
        distances = defaultdict(lambda: float('inf'))

        for m in matches:
            # check that the match is the best for each key point
            if m.distance > min(distances[m.trainIdx], self.distance):
                continue

            # check that the keypoint found is where we expected it to be
            pt1 = projection[m.queryIdx]
            pt2 = self.keypoints[m.trainIdx]
            dx = pt1[0] - pt2[0]
            dy = pt1[1] - pt2[1]
            if np.sqrt(dx*dx + dy*dy) > self.neighborhood:
                continue

            good_matches.append([m.queryIdx, m.trainIdx])
            distances[m.trainIdx] = m.distance

        return np.array(good_matches)

    def find_line_matches(self, lines, descriptors):

        matches = self.matcher.match(np.array(descriptors), self.line_descriptors)

        good_matches = []

        for m in matches:
            # check that the match is the best for each key point
            if m.distance > self.line_distance:
                continue

            good_matches.append([m.queryIdx, m.trainIdx])
        
        return np.array(good_matches)

    def get_feature(self, feature_type, i):
        if feature_type == FeatureType.Point:
            return self.keypoints[i]
        elif feature_type == FeatureType.Line:
            return self.keylines[i]
        raise ValueError(f'{feature_type} is not a valid feature type')

    def set_matched(self, feature_type, i):
        if feature_type == FeatureType.Point:
            self.unmatched_points[i] = False
        elif feature_type == FeatureType.Line:
            self.unmatched_lines[i] = False
        else:
            raise ValueError(f'{feature_type} is not a valid feature type')

    def get_descriptor(self, feature_type, i):
        if feature_type == FeatureType.Point:
            return self.descriptors[i]
        elif feature_type == FeatureType.Line:
            return self.line_descriptors[i]
        raise ValueError(f'{feature_type} is not a valid feature type')

    def get_color(self, kps, img):
        colors = []
        for kp in kps:
            x = int(np.clip(kp[0], 0, self.width-1))
            y = int(np.clip(kp[1], 0, self.height-1))
            colors.append(img[y, x] / 255)
        return colors


class StereoFrame:

    def __init__(self, left_frame, right_frame):
        self.left = left_frame
        self.right = right_frame

    def transform(self, points):    # from world coordinates
        return self.left.transform(points)

    def create_measurements_from_matches(self, matches_left, matches_right, map_primitives, feature_type):

        if feature_type == FeatureType.Point:
            Measurement = PointMeasurement
        elif feature_type == FeatureType.Line:
            Measurement = LineMeasurement

        measurements = []
        for mappoint_id, left_id in matches_left.items():
            # if match in both left and right
            if mappoint_id in matches_right:
                right_id = matches_right[mappoint_id]

                meas = Measurement(
                    MeasurementType.STEREO,
                    MeasurementSource.TRACKING,
                    map_primitives[mappoint_id],
                    [self.left.get_feature(feature_type, left_id),
                        self.right.get_feature(feature_type, right_id)],
                    [self.left.get_descriptor(feature_type, left_id),
                        self.right.get_descriptor(feature_type, right_id)])
                measurements.append(meas)
                self.left.set_matched(feature_type, left_id)
                self.right.set_matched(feature_type, right_id)

            # if only left is matched
            else:
                meas = Measurement(
                    MeasurementType.LEFT,
                    MeasurementSource.TRACKING,
                    map_primitives[mappoint_id],
                    [self.left.get_feature(feature_type, left_id)],
                    [self.left.get_descriptor(feature_type, left_id)])
                measurements.append(meas)
                self.left.set_matched(feature_type, mappoint_id)

        for mappoint_id, right_id in matches_right.items():
            # if only right is matched
            if mappoint_id not in matches_left:
                meas = Measurement(
                    MeasurementType.RIGHT,
                    MeasurementSource.TRACKING,
                    map_primitives[mappoint_id],
                    [self.right.get_feature(feature_type, right_id)],
                    [self.right.get_descriptor(feature_type, right_id)])
                measurements.append(meas)
                self.right.set_matched(feature_type, right_id)

        return measurements

    def visualise_measurements(self, measurements):

        img = np.array(self.left.image)
        for i, m in enumerate(measurements):
            if m.type != MeasurementType.RIGHT:
                left_keyline = m.keylines[0]

                pt1 = tuple(left_keyline[:2].astype(int))
                pt2 = tuple(left_keyline[2:].astype(int))
                c = m.mapline.color
                # c = (255, 0, 0)
                cv2.line(img,pt1,pt2,c,10)
        img = cv2.resize(img, (0,0), fx=0.5, fy=0.5) 
        cv2.imshow('left', img)

        # img = np.array(self.right.image)
        # for i, m in enumerate(good_matches):
        #     pt1 = tuple(self.right.keylines[m[1]][:2].astype(int))
        #     pt2 = tuple(self.right.keylines[m[1]][2:].astype(int))
        #     c = tuple([int(x) for x in colors[i]])
        #     cv2.line(img,pt1,pt2,c,2)
        # cv2.imshow('right', img)        
        cv2.waitKey(1)
        
    def match_maplines(self, maplines):
        lines = []
        descriptors = []
        for mapline in maplines:
            lines.append(mapline.endpoints)
            descriptors.append(mapline.descriptor)

        matches_left = dict(self.left.find_line_matches(lines, descriptors))
        matches_right = dict(self.right.find_line_matches(lines, descriptors))

        measurements = self.create_measurements_from_matches(matches_left, matches_right, maplines, FeatureType.Line)
        return measurements

    def match_mappoints(self, mappoints):
        points = []
        descriptors = []
        for mappoint in mappoints:
            points.append(mappoint.position)
            descriptors.append(mappoint.descriptor)

        matches_left = dict(self.left.find_matches(points, descriptors))
        matches_right = dict(self.right.find_matches(points, descriptors))

        measurements = self.create_measurements_from_matches(matches_left, matches_right, mappoints, FeatureType.Point)
        return measurements
    
    @property
    def image(self):
        return self.left.image

    @property
    def position(self):
        return self.left.pose.position()

    @property
    def orientation(self):
        return self.left.pose.orientation()

    @property
    def pose(self):
        return self.left.pose

    @property
    def params(self):
        return self.left.params

    @property
    def cam(self):
        return self.left.cam

    @property
    def idx(self):
        return self.left.idx

    @property
    def timestamp(self):
        return self.left.timestamp

    def update_pose(self, pose):
        self.left.update_pose(pose)
        self.right.update_pose(self.cam.compute_right_camera_pose(pose))

    def to_keyframe(self):
        return KeyFrame(self)


class KeyFrame(StereoFrame):
    _id = 0

    def __init__(self, stereo_frame):
        super().__init__(stereo_frame.left, stereo_frame.right)

        self.classes, _ = segmentation.segment_image(stereo_frame.left.image)

        self.meas = dict()

        self.id = KeyFrame._id
        KeyFrame._id += 1

        self.lock = Lock()

        self.preceding_keyframe = None
        self.loop_keyframe = None
        self.loop_constraint = None
        self.fixed = False

    def create_mappoints_from_triangulation(self):
        matches = self._match_key_points()
        points_3d = self._triangulate(matches)

        can_view = np.logical_and(
            self.left.can_view(points_3d),
            self.right.can_view(points_3d))

        points_3d = points_3d[can_view]
        matches = matches[can_view]

        mappoints = []
        for point, match in zip(points_3d, matches):
            normal = point - self.position
            normal = normal / np.linalg.norm(normal)
            mappoint = MapPoint(
                point, normal, self.left.descriptors[match[0]], self.left.colors[match[0]])
            mappoints.append(mappoint)

        measurements = []
        for mappoint, match in zip(mappoints, matches):

            seg_class = segmentation.find_seg_class(self.classes, self.left.keypoints[match[0]])

            meas = PointMeasurement(
                MeasurementType.STEREO,
                MeasurementSource.TRIANGULATION,
                mappoint,
                [self.left.keypoints[match[0]], self.right.keypoints[match[1]]],
                [self.left.descriptors[match[0]], self.right.descriptors[match[1]]],
                seg_class)
            meas.view = self.transform(mappoint.position)
            measurements.append(meas)

        return mappoints, measurements

    def create_maplines_from_triangulation(self):
        matches = self._match_key_lines()
        lines_3d, good = self._triangulate_lines(matches)

        # First good triangulation filter
        matches = matches[good]
        lines_3d = lines_3d[good]

        can_view = (self.left.can_view(lines_3d[:,:3]) *
            self.left.can_view(lines_3d[:,3:]) *
            self.right.can_view(lines_3d[:,:3]) *
            self.right.can_view(lines_3d[:,3:]))

        print(f'can view ratio: {np.sum(can_view) / len(can_view)}')

        # Then can view filter
        lines_3d = lines_3d[can_view]
        matches = matches[can_view]

        maplines = []
        for line, match in zip(lines_3d, matches):
            mapline = MapLine(line, self.left.line_descriptors[match[0]])
            maplines.append(mapline)

        measurements = []
        for mapline, match in zip(maplines, matches):
            meas = LineMeasurement(
                MeasurementType.STEREO,
                MeasurementSource.TRIANGULATION,
                mapline,
                [self.left.keylines[match[0]], self.right.keylines[match[1]]],
                [self.left.line_descriptors[match[0]], self.right.line_descriptors[match[1]]])
            measurements.append(meas)

        return maplines, measurements

    def _match_key_points(self, matching_distance=40, max_row_distance=2.5, max_disparity=100):
        matches = self.params.descriptor_matcher.match(self.left.descriptors, self.right.descriptors)
        assert len(matches) > 0

        good_count = 0
        good_matches = []
        for m in matches:
            pt1 = self.left.keypoints[m.queryIdx]
            pt2 = self.right.keypoints[m.trainIdx]
            if (m.distance < matching_distance and 
                abs(pt1[1] - pt2[1]) < max_row_distance and 
                abs(pt1[0] - pt2[0]) < max_disparity):  # epipolar constraint
                    good_count += 1

                    if self.left.unmatched_points[m.queryIdx] or self.right.unmatched_points[m.trainIdx]:  
                        good_matches.append([m.queryIdx, m.trainIdx])

        # print(f'New keyframe has {good_count} good points, of which {len(good_matches)} are unmatched')

        return np.array(good_matches)

    def _match_key_lines(self, matching_distance=30, min_length=10, length_ratio=0.9, min_y_disparity=10):

        matches = self.params.descriptor_matcher.match(self.left.line_descriptors, self.right.line_descriptors)
        assert len(matches) > 0

        def length(kl):
            return math.sqrt((kl[0] - kl[2]) ** 2 + (kl[1] - kl[3]) ** 2)

        good_count = 0
        good_matches = []
        for m in matches:
            kl1 = self.left.keylines[m.queryIdx]
            kl2 = self.right.keylines[m.trainIdx]
            l1 = length(kl1)
            l2 = length(kl2)
            if (m.distance < matching_distance and
                abs(kl1[1] - kl1[3]) > min_y_disparity and
                abs(kl2[1] - kl2[3]) > min_y_disparity and
                min(l1, l2) > min_length and
                min(l1, l2) / max(l1, l2) > length_ratio):
                good_count += 1

                if self.left.unmatched_lines[m.queryIdx] or self.right.unmatched_lines[m.trainIdx]:  
                    good_matches.append([m.queryIdx, m.trainIdx])

        # print(f'New keyframe has {good_count} good lines, of which {len(good_matches)} are unmatched')

        return np.array(good_matches)  

    def _triangulate(self, matches):
        pts_left = np.array([self.left.keypoints[m] for m in matches[:,0]])
        pts_right = np.array([self.right.keypoints[m] for m in matches[:,1]])

        points = cv2.triangulatePoints(
            self.left.projection_matrix,
            self.right.projection_matrix,
            pts_left.transpose(),
            pts_right.transpose()
        ).transpose()  # shape: (N, 4)
        points = points[:, :3] / points[:, 3:]

        return points  

    def _triangulate_lines(self, matches):
        kls_left = np.array([self.left.keylines[m] for m in matches[:,0]])
        kls_right = np.array([self.right.keylines[m] for m in matches[:,1]])

        lines, good = triangulate_lines(kls_left, kls_right,
            self.left.transform_matrix,
            self.right.transform_matrix,
            self.cam)

        return lines, good

    def add_measurement(self, m):
        self.meas[m] = m.get_map_primitive()

    def measurements(self):
        return self.meas.keys()

    def point_measurements(self):
        return [m for m in self.meas.keys() if m.is_point()]

    def mappoints(self):
        return [v for v in self.meas.values() if v.is_point()]

    def maplines(self):
        return [v for v in self.meas.values() if v.is_line()]

    def update_preceding(self, preceding):
        self.preceding_keyframe = preceding

    def get_preceding(self):
        return self.preceding_keyframe

    def is_fixed(self):
        return self.fixed

    def set_fixed(self, fixed=True):
        self.fixed = fixed
