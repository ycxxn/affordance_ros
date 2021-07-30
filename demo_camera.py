# -*- coding: utf-8 -*-
'''
@Time          : 20/04/25 15:49
@Author        : huguanghao
@File          : demo.py
@Noice         :
@Modificattion :
    @Author    :
    @Time      :
    @Detail    :
'''

# import sys
# import time
# from PIL import Image, ImageDraw
# from models.tiny_yolo import TinyYoloNet
from torch._C import unify_type_list
from tool.utils import *
from tool.torch_utils import *
from tool.darknet2pytorch import Darknet
import argparse
import numpy as np 
import cv2
"""hyper parameters"""
use_cuda = True

def detect_cv2_camera(cfgfile, weightfile):
    m = Darknet(cfgfile)

    m.print_network()
    m.load_weights(weightfile)
    print('Loading weights from %s... Done!' % (weightfile))

    if use_cuda:
        m.cuda()

    cap = cv2.VideoCapture(0)
    # cap = cv2.VideoCapture("video/video2.mp4")
    cap.set(3, 1280)
    cap.set(4, 720)
    print("Starting the YOLO loop...")

    # video_FourCC = cv2.VideoWriter_fourcc(*"mp4v")
    # out_video = cv2.VideoWriter("demo.mp4", video_FourCC, 30, (960, 540))
    namesfile = 'data/obj.names'
    
    class_names = load_class_names(namesfile)
    i = 1
    while True:
        ret, img = cap.read()
        sized = cv2.resize(img, (m.width, m.height))
        sized = cv2.cvtColor(sized, cv2.COLOR_BGR2RGB)

        start = time.time()
        boxes = do_detect(m, sized, 0.20, 0.6, use_cuda)
        finish = time.time()
        print('Predicted in %f seconds.' % (finish - start))

        result_img = plot_boxes_cv2(img, boxes[0], savename=None, class_names=class_names)
        print(boxes[0])
        # out_video.write(result_img)
        cv2.imshow('Yolo demo', result_img)
        cv2.waitKey(1)
    out_video.release()
    cap.release()


def get_args():
    parser = argparse.ArgumentParser('Test your image or video by trained model.')
    parser.add_argument('-cfgfile', type=str, default='./cfg/yolov4.cfg',
                        help='path of cfg file', dest='cfgfile')
    parser.add_argument('-weightfile', type=str,
                        default='./checkpoints/Yolov4_epoch1.pth',
                        help='path of trained model.', dest='weightfile')
    parser.add_argument('-imgfile', type=str,
                        default='./data/mscoco2017/train2017/190109_180343_00154162.jpg',
                        help='path of your image file.', dest='imgfile')
    args = parser.parse_args()

    return args


if __name__ == '__main__':
    args = get_args()
    detect_cv2_camera(args.cfgfile, args.weightfile)