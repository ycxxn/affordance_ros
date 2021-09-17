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
from std_msgs.msg import Float32MultiArray
# from rospy_tutorials.msg import Floats
from rospy.numpy_msg import numpy_msg
#detection
from torch._C import dtype, unify_type_list
from tool.utils import *
from tool.torch_utils import *
from tool.darknet2pytorch import Darknet

## segementation
# from modeling.deeplab import *
# from dataloaders import custom_transforms as tr
# from PIL import Image
# from torchvision import transforms
# from dataloaders.utils import  *
# from torchvision.utils import make_grid, save_image
# """hyper parameters"""
use_cuda = False #True


def split_object(boxes):
    popcorn = []
    juice = []

    #bb[0~6]: x1, y1, x2, y2, ?, confidence, class_id
    for bb in boxes:
        if bb[6] == 0 or bb[6]==1:
            popcorn.append(bb)
        if bb[6] == 2:
            juice.append(bb)
    
    return popcorn, juice


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
        self.init_yolo(namesfile, cfg, w)
        rospy.Subscriber("/camera/color/image_raw", msg_Image, self.imageCallback)
        self.popcorn_pub = rospy.Publisher('popcorn_pub', Float32MultiArray, queue_size=10)
        self.juice_pub = rospy.Publisher('juice_pub', Float32MultiArray, queue_size=10)
        
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
        self.cv_image = cv2.resize(cv_image, (640,480))
        self.det()

    def pub(self, popcorn, juice):
        juice = np.array(juice, dtype=np.float32).flatten()
        popcorn = np.array(popcorn, dtype=np.float32).flatten()

        juice_data = Float32MultiArray(data = juice)
        popcorn_data = Float32MultiArray(data = popcorn)
        
        self.juice_pub.publish(juice_data) 
        self.popcorn_pub.publish(popcorn_data)       
              
    def det(self):
        sized = cv2.resize(self.cv_image, (self.m.width, self.m.height))
        sized = cv2.cvtColor(sized, cv2.COLOR_BGR2RGB)

        boxes = do_detect(self.m, sized, 0.20, 0.6, use_cuda)
        result_img = plot_boxes_cv2(self.cv_image, boxes[0], savename=None, class_names=self.class_names)

        popcorn, juice = split_object(boxes[0])
        self.pub(popcorn, juice)

        print('===============================')
        print("popcorn: {}".format(popcorn))
        print("juice: {}".format(juice))

        cv2.imshow("result_img", result_img)
        cv2.waitKey(1)

    

if __name__ == '__main__':
    args = get_args()    
    Node(args.namesfile, args.cfgfile, args.weightfile)

   