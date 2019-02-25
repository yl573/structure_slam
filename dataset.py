import numpy as np
import cv2
import os
import time

from collections import defaultdict, namedtuple

from threading import Thread, Lock
from multiprocessing import Process, Queue



class ImageReader(object):
    def __init__(self, ids, timestamps, cam=None):
        self.ids = ids
        self.timestamps = timestamps
        self.cam = cam
        self.cache = dict()
        self.idx = 0

        self.ahead = 10      # 10 images ahead of current index
        self.waiting = 1.5   # waiting time

        self.preload_thread = Thread(target=self.preload)
        self.thread_started = False

    def read(self, path):
        img = cv2.imread(path, -1)
        if self.cam is None:
            return img
        else:
            return self.cam.rectify(img)
        
    def preload(self):
        idx = self.idx
        t = float('inf')
        while True:
            if time.time() - t > self.waiting:
                return
            if self.idx == idx:
                time.sleep(1e-2)
                continue
            
            for i in range(self.idx, self.idx + self.ahead):
                if i not in self.cache and i < len(self.ids):
                    self.cache[i] = self.read(self.ids[i])
            if self.idx + self.ahead > len(self.ids):
                return
            idx = self.idx
            t = time.time()
    
    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        self.idx = idx
        # if not self.thread_started:
        #     self.thread_started = True
        #     self.preload_thread.start()

        if idx in self.cache:
            img = self.cache[idx]
            del self.cache[idx]
        else:   
            img = self.read(self.ids[idx])
        return img

    def __iter__(self):
        for i, timestamp in enumerate(self.timestamps):
            yield timestamp, self[i]

    @property
    def dtype(self):
        return self[0].dtype
    @property
    def shape(self):
        return self[0].shape




class KITTIOdometry(object):   # without lidar
    '''
    path example: 'path/to/your/KITTI odometry dataset/sequences/00'
    '''
    def __init__(self, path):
        Cam = namedtuple('cam', 'fx fy cx cy width height baseline')
        cam00_02 = Cam(718.856, 718.856, 607.1928, 185.2157, 1241, 376, 0.5371657)
        cam03 = Cam(721.5377, 721.5377, 609.5593, 172.854, 1241, 376, 0.53715)
        cam04_12 = Cam(707.0912, 707.0912, 601.8873, 183.1104, 1241, 376, 0.53715)

        path = os.path.expanduser(path)
        timestamps = np.loadtxt(os.path.join(path, 'times.txt'))
        self.left = ImageReader(self.listdir(os.path.join(path, 'image_2')), 
            timestamps)
        self.right = ImageReader(self.listdir(os.path.join(path, 'image_3')), 
            timestamps)

        assert len(self.left) == len(self.right)
        self.timestamps = self.left.timestamps

        sequence = int(path.strip(os.path.sep).split(os.path.sep)[-1])
        if sequence < 3:
            self.cam = cam00_02
        elif sequence == 3:
            self.cam = cam03
        elif sequence < 13:
            self.cam = cam04_12

    def sort(self, xs):
        return sorted(xs, key=lambda x:float(x[:-4]))

    def listdir(self, dir):
        files = [_ for _ in os.listdir(dir) if _.endswith('.png')]
        return [os.path.join(dir, _) for _ in self.sort(files)]

    def __len__(self):
        return len(self.left)


class_to_trainid = {
    'Road': 0,
    'GuardRail': 1,
    'Building': 2,
    'Tree': 4,
    'Pole': 5,
    'TrafficLight': 6,
    'TrafficSign': 7,
    'Vegetation': 8,
    'Terrain': 9, 
    'Sky': 10,
    'Car': 13,
    'Truck': 14,
    'Van': 15,
    'Misc': 255
}


# if floyd:
#     rgb_path = os.path.join(cfg.DATA.FLOYD_DATA_PATH, 'vkitti_1.3.1_rgb')
#     seg_path = os.path.join(cfg.DATA.FLOYD_DATA_PATH, 'vkitti_1.3.1_scenegt')
# else:
#     rgb_path = os.path.join(cfg.DATA.DATA_PATH, 'vkitti_1.3.1_rgb')
#     seg_path = os.path.join(cfg.DATA.DATA_PATH, 'vkitti_1.3.1_scenegt')


def get_img_seq(path):
    img_paths = []
    img_metadata = []
    for subdir in os.listdir(path):
        subdir_path = os.path.join(rgb_path, subdir)
        if not os.path.isdir(subdir_path):
            continue
        for subtype in os.listdir(subdir_path):
            subtype_path = os.path.join(subdir_path, subtype)
            if not os.path.isdir(subtype_path):
                continue
            for img in os.listdir(subtype_path):
                img_path = os.path.join(subtype_path, img)
                img_paths.append(img_path)
                metadata = (subdir, subtype, img)
                img_metadata.append(metadata)

    seg_rbg_paths = []
    for subdir, subtype, img in img_metadata:
        seg_rbg_path = os.path.join(seg_path, subdir, subtype, img)
        if os.path.isfile(seg_rbg_path):
            seg_rbg_paths.append(seg_rbg_path)  

    if mode == 'train':
        return list(zip(img_paths[:20000], seg_rbg_paths[:20000]))
    if mode == 'val':
        return list(zip(img_paths[20000:], seg_rbg_paths[20000:]))
    return ValueError()

def parse_encoding(content):
    parsed = {}
    for c in content[1:]:
        terms = c.split(' ')
        seg_class = terms[0]
        if 'Car' in seg_class:
            seg_class = 'Car'
        if 'Van' in seg_class:
            seg_class = 'Van'
        key = class_to_trainid[seg_class]
        terms[1:] = [int(t) for t in terms[1:]]
        val = torch.Tensor(terms[1:]).long()
        parsed[key] = val
    return parsed

def make_encodings():
    encodings = {}
    encoding_files = [p for p in os.listdir(seg_path) if ('.txt' in p and 'README' not in p)]
    for encoding_file in encoding_files:
        sub_names = encoding_file.split('_')
        key = f'{sub_names[0]}_{sub_names[1]}'
        encoding_path = os.path.join(seg_path, encoding_file)
        with open(encoding_path, 'r') as f:
            content = f.readlines()
        encodings[key] = parse_encoding(content)
    return encodings

class VKITTIOdometry:

    def __init__(self, path):
        Cam = namedtuple('cam', 'fx fy cx cy width height baseline')
        self.cam = Cam(725, 725, 620.5, 187.0, 1241, 376, 0.53715)

        self.imgs = get_img_seq(path)

        path = os.path.expanduser(path)
        timestamps = np.loadtxt(os.path.join(path, 'times.txt'))
        self.left = ImageReader(self.listdir(os.path.join(path, 'image_2')), 
            timestamps)
        self.right = ImageReader(self.listdir(os.path.join(path, 'image_3')), 
            timestamps)

        assert len(self.left) == len(self.right)
        self.timestamps = self.left.timestamps

        sequence = int(path.strip(os.path.sep).split(os.path.sep)[-1])

    def sort(self, xs):
        return sorted(xs, key=lambda x:float(x[:-4]))

    def listdir(self, dir):
        files = [_ for _ in os.listdir(dir) if _.endswith('.png')]
        return [os.path.join(dir, _) for _ in self.sort(files)]

    def __len__(self):
        return len(self.left)






class Camera(object):
    def __init__(self, 
            width, height,
            intrinsic_matrix, 
            undistort_rectify=False,
            extrinsic_matrix=None,
            distortion_coeffs=None,
            rectification_matrix=None,
            projection_matrix=None):

        self.width = width
        self.height = height
        self.intrinsic_matrix = intrinsic_matrix
        self.extrinsic_matrix = extrinsic_matrix
        self.distortion_coeffs = distortion_coeffs
        self.rectification_matrix = rectification_matrix
        self.projection_matrix = projection_matrix
        self.undistort_rectify = undistort_rectify
        self.fx = intrinsic_matrix[0, 0]
        self.fy = intrinsic_matrix[1, 1]
        self.cx = intrinsic_matrix[0, 2]
        self.cy = intrinsic_matrix[1, 2]

        if undistort_rectify:
            self.remap = cv2.initUndistortRectifyMap(
                cameraMatrix=self.intrinsic_matrix,
                distCoeffs=self.distortion_coeffs,
                R=self.rectification_matrix,
                newCameraMatrix=self.projection_matrix,
                size=(width, height),
                m1type=cv2.CV_8U)
        else:
            self.remap = None

    def rectify(self, img):
        if self.remap is None:
            return img
        else:
            return cv2.remap(img, *self.remap, cv2.INTER_LINEAR)

class StereoCamera(object):
    def __init__(self, left_cam, right_cam):
        self.left_cam = left_cam
        self.right_cam = right_cam

        self.width = left_cam.width
        self.height = left_cam.height
        self.intrinsic_matrix = left_cam.intrinsic_matrix
        self.extrinsic_matrix = left_cam.extrinsic_matrix
        self.fx = left_cam.fx
        self.fy = left_cam.fy
        self.cx = left_cam.cx
        self.cy = left_cam.cy
        self.baseline = abs(right_cam.projection_matrix[0, 3] / 
            right_cam.projection_matrix[0, 0])
        self.focal_baseline = self.fx * self.baseline