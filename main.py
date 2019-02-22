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

from utils import RunningAverageTimer


def main(args):
    dataset = KITTIOdometry(args.path)
    params = ParamsKITTI()
    cam = Camera(
        dataset.cam.fx, dataset.cam.fy, dataset.cam.cx, dataset.cam.cy, 
        dataset.cam.width, dataset.cam.height, 
        params.frustum_near, params.frustum_far, 
        dataset.cam.baseline)

    tracker = Tracker(params, cam)
    viewer = MapViewer(tracker, params)

    if args.view_map:
        viewer.load_points()
        return

    timer = RunningAverageTimer()
    
    durations = []

    try:
        for i in range(len(dataset)):
            j = i + 0

            # Data loading takes 0.036s
            left = dataset.left[j]
            right = dataset.right[j]
            
            tracker.update(i, left, right, timestamp=dataset.timestamps[j])
            
            # Viewer update takes 0.002s
            viewer.update(refresh=True)

            print(f'Frame {j}')

            # input("Press any key to continue...")
            # import time
            # time.sleep(2)
    except:
        pass
    finally:
        viewer.stop()
        viewer.save_points()

if __name__ == '__main__':
    args = AttrDict(path='/Users/yuxuanliu/Desktop/kitti/00/')
    args.view_map = True
    main(args)