import numpy as np
import cv2
import g2o

from threading import Lock, Thread
from queue import Queue

from enum import Enum
from collections import defaultdict
from numbers import Number
from map_point import MapPoint, Measurement


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
        
        self.orientation = pose.orientation()
        self.position = pose.position()
        self.transform_matrix = pose.inverse().matrix()[:3]  # shape: (3, 4)
        self.projection_matrix = (
            self.cam.intrinsic.dot(self.transform_matrix))  # from world frame to image

        self.image = image
        self.height, self.width = image.shape[:2]

        self.keypoints = self.detector.detect(image)
        self.keypoints, self.descriptors = self.extractor.compute(image, self.keypoints)
        self.colors = self.get_color(self.keypoints, image)

    # batch version

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

    def update_pose(self, pose):
        if isinstance(pose, g2o.SE3Quat):
            self.pose = g2o.Isometry3d(pose.orientation(), pose.position())
        else:
            self.pose = pose
        self.orientation = self.pose.orientation()
        self.position = self.pose.position()

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
        proj, _ = self.project(self.transform(points))
        proj = proj.transpose()
        return self.find_matches_feature(proj, descriptors)

    def find_matches_feature(self, predictions, descriptors):
        matches = dict()
        distances = defaultdict(lambda: float('inf'))
        for m, query_idx, train_idx in self.matched_by(descriptors):
            if m.distance > min(distances[train_idx], self.distance):
                continue

            pt1 = predictions[query_idx]
            pt2 = self.keypoints[train_idx].pt
            dx = pt1[0] - pt2[0]
            dy = pt1[1] - pt2[1]
            if np.sqrt(dx*dx + dy*dy) > self.neighborhood:
                continue

            matches[train_idx] = query_idx
            distances[train_idx] = m.distance
        matches = [(i, j) for j, i in matches.items()]
        return matches

    def matched_by(self, descriptors):
        unmatched_descriptors = self.descriptors
        if len(unmatched_descriptors) == 0:
            return []

        # TODO: reduce matched points by using predicted position
        matches = self.matcher.match(
            np.array(descriptors), unmatched_descriptors)
        return [(m, m.queryIdx, m.trainIdx) for m in matches]

    def get_keypoint(self, i):
        return self.keypoints[i]

    def get_descriptor(self, i):
        return self.descriptors[i]

    def get_color(self, kps, img):
        colors = []
        for kp in kps:
            x = int(np.clip(kp.pt[0], 0, self.width-1))
            y = int(np.clip(kp.pt[1], 0, self.height-1))
            colors.append(img[y, x] / 255)
        return colors


class StereoFrame:

    def __init__(self, left_frame, right_frame):
        self.left = left_frame
        self.right = right_frame

    def transform(self, points):    # from world coordinates
        return self.left.transform(points)

    def match_mappoints(self, mappoints, source):

        points = []
        descriptors = []
        for mappoint in mappoints:
            points.append(mappoint.position)
            descriptors.append(mappoint.descriptor)

        matches_left = dict(self.left.find_matches(points, descriptors))
        matches_right = dict(self.right.find_matches(points, descriptors))

        measurements = []
        for i, j in matches_left.items():
            # if match in both left and right
            if i in matches_right:
                j2 = matches_right[i]

                y1 = self.left.get_keypoint(j).pt[1]
                y2 = self.right.get_keypoint(j2).pt[1]
                if abs(y1 - y2) > 2.5:    # epipolar constraint
                    continue   # TODO: choose one

                meas = Measurement(
                    Measurement.Type.STEREO,
                    source,
                    mappoints[i],
                    [self.left.get_keypoint(j),
                        self.right.get_keypoint(j2)],
                    [self.left.get_descriptor(j),
                        self.right.get_descriptor(j2)])
                measurements.append(meas)

            # if only left is matched
            else:
                meas = Measurement(
                    Measurement.Type.LEFT,
                    source,
                    mappoints[i],
                    [self.left.get_keypoint(j)],
                    [self.left.get_descriptor(j)])
                measurements.append(meas)

        for i, j in matches_right.items():
            # if only right is matched
            if i not in matches_left:
                meas = Measurement(
                    Measurement.Type.RIGHT,
                    source,
                    mappoints[i],
                    [self.right.get_keypoint(j)],
                    [self.right.get_descriptor(j)])
                measurements.append(meas)

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
    _id_lock = Lock()

    def __init__(self, stereo_frame):
        # super().__init__(frame.idx, frame.pose, frame.cam, frame.params, frame/, img_right,
        #          right_cam=None, timestamp=None)

        super().__init__(stereo_frame.left, stereo_frame.right)

        self.meas = dict()

        with KeyFrame._id_lock:
            self.id = KeyFrame._id
            KeyFrame._id += 1

        self.preceding_keyframe = None
        self.preceding_constraint = None
        self.loop_keyframe = None
        self.loop_constraint = None
        self.fixed = False

    def transform(self, points):    # from world coordinates
        return self.left.transform(points)

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
            meas = Measurement(
                Measurement.Type.STEREO,
                Measurement.Source.TRIANGULATION,
                mappoint,
                [self.left.keypoints[match[0]], self.right.keypoints[match[1]]],
                [self.left.descriptors[match[0]], self.right.descriptors[match[1]]])
            meas.view = self.transform(mappoint.position)
            measurements.append(meas)

        return mappoints, measurements

    def _triangulate(self, matches):
        pts_left = np.array([self.left.keypoints[m].pt for m in matches[:,0]])
        pts_right = np.array([self.right.keypoints[m].pt for m in matches[:,1]])

        points = cv2.triangulatePoints(
            self.left.projection_matrix,
            self.right.projection_matrix,
            pts_left.transpose(),
            pts_right.transpose()
        ).transpose()  # shape: (N, 4)
        points = points[:, :3] / points[:, 3:]

        return points

    def _match_key_points(self, matching_distance=40, max_row_distance=2.5, max_disparity=100):

        matches = self.params.descriptor_matcher.match(self.left.descriptors, self.right.descriptors)
        assert len(matches) > 0

        good_matches = []
        for m in matches:
            pt1 = self.left.keypoints[m.queryIdx].pt
            pt2 = self.right.keypoints[m.trainIdx].pt
            if (m.distance < matching_distance and 
                abs(pt1[1] - pt2[1]) < max_row_distance and 
                abs(pt1[0] - pt2[0]) < max_disparity):   # epipolar constraint
                good_matches.append([m.queryIdx, m.trainIdx])
        return np.array(good_matches)

    def add_measurement(self, m):
        self.meas[m] = m.mappoint

    def measurements(self):
        return self.meas.keys()

    def mappoints(self):
        return self.meas.values()

    def update_preceding(self, preceding=None):
        if preceding is not None:
            self.preceding_keyframe = preceding
        self.preceding_constraint = (
            self.preceding_keyframe.pose.inverse() * self.pose)

    def is_fixed(self):
        return self.fixed

    def set_fixed(self, fixed=True):
        self.fixed = fixed
