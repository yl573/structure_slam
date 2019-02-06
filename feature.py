import numpy as np
import cv2

from collections import defaultdict
from numbers import Number

from threading import Thread, Lock 
from queue import Queue



class ImageFeature(object):
    def __init__(self, image, params):
        # TODO: pyramid representation
        self.image = image 
        self.height, self.width = image.shape[:2]

        self.keypoints = []      # list of cv2.KeyPoint
        self.descriptors = []    # numpy.ndarray

        self.detector = params.feature_detector
        self.extractor = params.descriptor_extractor
        self.matcher = params.descriptor_matcher

        self.cell_size = params.matching_cell_size
        self.distance = params.matching_distance
        self.neighborhood = (
            params.matching_cell_size * params.matching_neighborhood)