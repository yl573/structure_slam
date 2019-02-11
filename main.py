import cv2

import os
import sys
import argparse

from threading import Thread
from attrdict import AttrDict

from camera import Camera
from params import ParamsKITTI, ParamsEuroc
from dataset import KITTIOdometry, EuRoCDataset
from tracker import Tracker
from viewer import MapViewer
import time

from viewer import MapViewer

def main(args):
    dataset = KITTIOdometry(args.path)
    params = ParamsKITTI()
    cam = Camera(
        dataset.cam.fx, dataset.cam.fy, dataset.cam.cx, dataset.cam.cy, 
        dataset.cam.width, dataset.cam.height, 
        params.frustum_near, params.frustum_far, 
        dataset.cam.baseline)

    # shared_map = Map()
    tracker = Tracker(params, cam)
    # viewer = MapViewer(shared_map)
    viewer = MapViewer(tracker, params)
    
    durations = []
    for i in range(len(dataset)):
        j = i + 10
        tracker.update(i, dataset.left[j], dataset.right[j], timestamp=dataset.timestamps[j])
        viewer.update()

        # import time
        # time.sleep(20)
    
    viewer.stop()

if __name__ == '__main__':
    args = AttrDict(path='/Volumes/MyPassport/KITTI/dataset/sequences/00/')
    main(args)