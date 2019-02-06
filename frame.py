import numpy as np
import cv2
import g2o

from threading import Lock, Thread
from queue import Queue

from enum import Enum
from collections import defaultdict
from numbers import Number


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
        self.height, self.width = self.image.shape[:2]

        self.keypoints = self.detector.detect(self.image)
        self.keypoints, self.descriptors = self.extractor.compute(
            self.image, self.keypoints)

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

    def get_color(self, pt):
        x = int(np.clip(pt[0], 0, self.width-1))
        y = int(np.clip(pt[1], 0, self.height-1))
        color = self.image[y, x]
        if isinstance(color, Number):
            color = np.array([color, color, color])
        return color[::-1] / 255.

    def get_unmatched_keypoints(self):
        return self.keypoints, self.descriptors, list(range(len(self.keypoints)))


class StereoFrame(Frame):
    def __init__(self, idx, pose, cam, params, img_left, img_right,
                 right_cam=None, timestamp=None):

        super().__init__(idx, pose, cam, params, img_left, timestamp)
        self.left = Frame(idx, pose, cam, params, img_left, timestamp)
        self.right = Frame(idx,
                           cam.compute_right_camera_pose(pose),
                           right_cam or cam,
                           params, img_right, timestamp)


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

    def triangulate(self):
        kps_left, desps_left, idx_left = self.left.get_unmatched_keypoints()
        kps_right, desps_right, idx_right = self.right.get_unmatched_keypoints()

        mappoints, matches = self.triangulate_points(
            kps_left, desps_left, kps_right, desps_right)

        measurements = []
        for mappoint, (i, j) in zip(mappoints, matches):
            meas = Measurement(
                Measurement.Type.STEREO,
                Measurement.Source.TRIANGULATION,
                mappoint,
                [kps_left[i], kps_right[j]],
                [desps_left[i], desps_right[j]])
            meas.view = self.transform(mappoint.position)
            measurements.append(meas)

        return mappoints, measurements

    def row_match(self, kps1, desps1, kps2, desps2,
            matching_distance=40, 
            max_row_distance=2.5, 
            max_disparity=100):

        matches = self.matcher.match(np.array(desps1), np.array(desps2))
        good = []
        for m in matches:
            pt1 = kps1[m.queryIdx].pt
            pt2 = kps2[m.trainIdx].pt
            if (m.distance < matching_distance and 
                abs(pt1[1] - pt2[1]) < max_row_distance and 
                abs(pt1[0] - pt2[0]) < max_disparity):   # epipolar constraint
                good.append(m)
        return good

    def triangulate_points(self, kps_left, desps_left, kps_right, desps_right):
        matches = self.row_match(
            kps_left, desps_left, kps_right, desps_right)
        assert len(matches) > 0

        px_left = np.array([kps_left[m.queryIdx].pt for m in matches])
        px_right = np.array([kps_right[m.trainIdx].pt for m in matches])

        points = cv2.triangulatePoints(
            self.left.projection_matrix,
            self.right.projection_matrix,
            px_left.transpose(),
            px_right.transpose()
        ).transpose()  # shape: (N, 4)

        points = points[:, :3] / points[:, 3:]

        can_view = np.logical_and(
            self.left.can_view(points),
            self.right.can_view(points))

        mappoints = []
        matchs = []
        for i, point in enumerate(points):
            if not can_view[i]:
                continue
            normal = point - self.position
            normal = normal / np.linalg.norm(normal)

            color = self.left.get_color(px_left[i])

            mappoint = MapPoint(
                point, normal, desps_left[matches[i].queryIdx], color)
            mappoints.append(mappoint)
            matchs.append((matches[i].queryIdx, matches[i].trainIdx))

        return mappoints, matchs

    def update_pose(self, pose):
        super().update_pose(pose)
        self.right.update_pose(pose)
        self.left.update_pose(
            self.cam.compute_right_camera_pose(pose))

    # batch version
    def can_view(self, mappoints):
        points = []
        point_normals = []
        for i, p in enumerate(mappoints):
            points.append(p.position)
            point_normals.append(p.normal)
        points = np.asarray(points)
        point_normals = np.asarray(point_normals)

        normals = points - self.position
        normals /= np.linalg.norm(normals, axis=-1, keepdims=True)
        cos = np.clip(np.sum(point_normals * normals, axis=1), -1, 1)
        parallel = np.arccos(cos) < (np.pi / 4)

        can_view = np.logical_or(
            self.left.can_view(points),
            self.right.can_view(points))

        return np.logical_and(parallel, can_view)

    def to_keyframe(self):
        return KeyFrame(
            self.idx, self.pose,
            self.cam, self.params, self.left.image, self.right.image, self.right.cam)


class KeyFrame(StereoFrame):
    _id = 0
    _id_lock = Lock()

    def __init__(self, *args, **kwargs):
        StereoFrame.__init__(self, *args, **kwargs)
        self.meas = dict()

        with KeyFrame._id_lock:
            self.id = KeyFrame._id
            KeyFrame._id += 1

        self.preceding_keyframe = None
        self.preceding_constraint = None
        self.loop_keyframe = None
        self.loop_constraint = None
        self.fixed = False

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

    def set_loop(self, keyframe, constraint):
        self.loop_keyframe = keyframe
        self.loop_constraint = constraint

    def is_fixed(self):
        return self.fixed

    def set_fixed(self, fixed=True):
        self.fixed = fixed


class MapPoint:
    _id = 0
    _id_lock = Lock()

    def __init__(self, position, normal, descriptor,
                 color=np.zeros(3),
                 covariance=np.identity(3) * 1e-4):

        with MapPoint._id_lock:
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
        super().__init__()

        self.mappoint = mappoint

        self.type = type
        self.source = source
        self.keypoints = keypoints
        self.descriptors = descriptors
        self.view = None    # mappoint's position in current coordinates frame

        self.xy = np.array(self.keypoints[0].pt)
        if self.is_stereo():
            self.xyx = np.array([
                *keypoints[0].pt, keypoints[1].pt[0]])

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
