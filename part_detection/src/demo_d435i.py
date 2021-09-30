#!/usr/bin/env python3  
# -*- coding: utf-8 -*-
import argparse
from matplotlib.pyplot import jet
import numpy as np 
import cv2
import rospy
import sys
sys.path.insert(1, '/opt/installer/open_cv/cv_bridge/lib/python3/dist-packages/')
from cv_bridge import CvBridge, CvBridgeError

from sensor_msgs.msg import Image as msg_Image
from part_detection.msg import yolo_bbox
from part_detection.msg import yolo_bboxes

#detection
from torch._C import dtype, unify_type_list
from tool.utils import *
from tool.torch_utils import *
from tool.darknet2pytorch import Darknet

use_cuda = False #True

def get_args():
    parser = argparse.ArgumentParser('Test your image or video by trained model.')
    parser.add_argument('-namesfile', type=str, default='./data/20210806/obj.names',
                        help='path of name file', dest='namesfile')
    parser.add_argument('-cfgfile', type=str, default='./cfg/20210806/yolov4.cfg',
                        help='path of cfg file', dest='cfgfile')
    parser.add_argument('-weightfile', type=str,
                        default='./weights/20210806/yolov4_final.weights',
                        help='path of trained model.', dest='weightfile')
    args, unknown = parser.parse_known_args()
    return args


class Node:
    def __init__(self, namesfile, cfg, w):
        rospy.init_node("yolov4_node")

        self.bridge = CvBridge()
        self.img_width = 640
        self.img_height = 480
        self.init_yolo(namesfile, cfg, w)
        rospy.Subscriber("/camera/color/image_raw", msg_Image, self.imageCallback)        
        self.popcorn_pub = rospy.Publisher('popcorn_pub', yolo_bboxes, queue_size=10)
        self.juice_pub = rospy.Publisher('juice_pub', yolo_bboxes, queue_size=10)

        rospy.spin()

    def init_yolo(self, namesfile, cfg, w):
        m = Darknet(cfg)
        m.print_network()
        m.load_weights(w)
        if use_cuda:
            m.cuda()
        self.m = m

        self.class_names = load_class_names(namesfile)

    def imageCallback(self, rgb):
        cv_image = self.bridge.imgmsg_to_cv2(rgb, "bgr8")
        self.cv_image = cv2.resize(cv_image, (self.img_width, self.img_height))
        self.det()
   
    def split_object(self, boxes):    
        popcorn_bboxes = yolo_bboxes()
        juice_bboxes = yolo_bboxes()
        popcorn_list = []
        juice_list = []

        for bb in boxes:
            box = yolo_bbox()
            box.min_x = int(bb[0] * self.img_width)
            box.min_y = int(bb[1] * self.img_height)
            box.max_x = int(bb[2] * self.img_width)
            box.max_y = int(bb[3] * self.img_height)
            box.score = bb[4] #bb[4] = bb[5]            
            box.object_name = self.class_names[bb[6]]

            if box.object_name == 'popcorn_f' or box.object_name == 'popcorn_b': #bb[6] == 0, 1
                popcorn_list.append(box)
            if box.object_name == 'juice': #bb[6] == 2
                juice_list.append(box)
        
        popcorn_bboxes.bboxes = popcorn_list
        juice_bboxes.bboxes = juice_list

        return popcorn_bboxes, juice_bboxes

    def det(self):
        sized = cv2.resize(self.cv_image, (self.m.width, self.m.height))
        sized = cv2.cvtColor(sized, cv2.COLOR_BGR2RGB)

        boxes = do_detect(self.m, sized, 0.20, 0.6, use_cuda)
        result_img = plot_boxes_cv2(self.cv_image, boxes[0], savename=None, class_names=self.class_names)

        popcorn_bboxes, juice_bboxes = self.split_object(boxes[0])
        
        self.popcorn_pub.publish(popcorn_bboxes)
        self.juice_pub.publish(juice_bboxes) 
        
        print('===============================')
        print("popcorn: {}".format(popcorn_bboxes))
        print("juice: {}".format(juice_bboxes))

        cv2.imshow("result_img", result_img)
        cv2.waitKey(1)   
    
if __name__ == '__main__':
    args = get_args()    
    Node(args.namesfile, args.cfgfile, args.weightfile)

   