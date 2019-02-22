import torch
from model.enet import ENet
from model.config import cfg
import numpy as np
from PIL import Image
import torchvision.transforms as standard_transforms
import torchvision.transforms.functional as F
import cv2

# from matplotlib import pyplot as plt

def class_color(id):
    return cfg.VIS.PALETTE_LABEL_COLORS[id]

class SegmentationModel:

    WIDTH = 1240
    HEIGHT = 376

    def __init__(self):

        self.net = ENet(only_encode=True)
        # # encoder_weight = torch.load('./model/encoder_ep_497_mIoU_0.5098.pth', map_location='cpu')
        encoder_weight = torch.load('/Users/yuxuanliu/Desktop/enet.pytorch/ckpt/kitti_checkpoint_19-02-20_07-59-33_encoder_ENet_city_[320, 640]_lr_0.0005.pth', map_location='cpu')
        self.net.encoder.load_state_dict(encoder_weight)     

        mean_std = cfg.DATA.MEAN_STD
        self.transform = standard_transforms.Compose([
            standard_transforms.ToTensor(),
            standard_transforms.Normalize(*mean_std)
        ])

        # self.test_model()


    # def test_model(self):
    #     img = torch.zeros([5, 3, 100, 100], dtype=torch.float)
    #     seg = self.net.forward(img).data

    def prepare_image(self, img):
        pil_img = Image.fromarray(img.astype(np.uint8))
        img_tensor = self.transform(pil_img)
        img_tensor = img_tensor[:3, :SegmentationModel.HEIGHT, :SegmentationModel.WIDTH]
        img_tensor = img_tensor.unsqueeze(0)

        return img_tensor

    def segment_image(self, img):  

        img_tensor = self.prepare_image(img) 

        seg = self.net.forward(img_tensor)
        max_vals, classes = torch.max(seg, 1)

        classes = classes.squeeze(0)

        np_classes = classes.data.numpy()
        color_mask = colorize_mask(np_classes)

        seg_img = np.asarray(color_mask)
        seg_img = Image.fromarray(seg_img.astype(np.uint8)).resize((img.shape[1], img.shape[0]))

        return np_classes, seg_img

        # plt.figure()
        # plt.imshow(seg_img)
        # plt.figure()
        # plt.imshow(Image.fromarray(img.astype(np.uint8)))
        # plt.show()

    def find_seg_class(self, class_mask, position):
        mask_height, mask_width = class_mask.shape
        sample_x = int(position[0] / SegmentationModel.WIDTH * mask_width)
        sample_y = int(position[1] / SegmentationModel.HEIGHT * mask_height)

        return class_mask[sample_y, sample_x]


def colorize_mask(mask):
    color_mask = np.zeros((*mask.shape, 3), dtype=np.int)
    for i in range(19):
        color_mask[mask==i] = cfg.VIS.PALETTE_LABEL_COLORS[i]

    return color_mask
        
        
