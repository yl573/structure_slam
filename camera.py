import numpy as np
import g2o

class Camera(object):
    def __init__(self, fx, fy, cx, cy, width, height, 
            frustum_near, frustum_far, baseline):
        self.fx = fx
        self.fy = fy
        self.cx = cx
        self.cy = cy
        self.baseline = baseline

        self.intrinsic = np.array([
            [fx, 0, cx], 
            [0, fy, cy], 
            [0, 0, 1]])

        self.frustum_near = frustum_near
        self.frustum_far = frustum_far

        self.width = width
        self.height = height
        
    def compute_right_camera_pose(self, pose):
        pos = pose * np.array([self.baseline, 0, 0])
        return g2o.Isometry3d(pose.orientation(), pos)