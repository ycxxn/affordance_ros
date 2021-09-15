#!/usr/bin/env python3  
import numpy as np 
import cv2
import rospy
import sys
sys.path.insert(0, '/opt/installer/open_cv/cv_bridge/lib/python3/dist-packages/')
from cv_bridge import CvBridge, CvBridgeError
import rospy
from sensor_msgs.msg import Image as msg_Image
import message_filters

class mask_all_Node:
    def __init__(self):
        rospy.init_node("show_all_seg")
        self.bridge = CvBridge()
        # rospy.Subscriber("popcorn_seg_img", msg_Image, self.imageCallback)
        # rospy.Subscriber("juice_seg_img", msg_Image, self.imageCallback)
        self.popcorn_sub = message_filters.Subscriber("popcorn_seg_img", msg_Image)
        self.juice_sub = message_filters.Subscriber("juice_seg_img", msg_Image)
        self.ts = message_filters.TimeSynchronizer([self.popcorn_sub, self.juice_sub],10)
        self.ts.registerCallback(self.callback)
        rospy.spin()

    def callback(self, img1, img2):
        img1 = self.bridge.imgmsg_to_cv2(img1, desired_encoding='passthrough')
        img2 = self.bridge.imgmsg_to_cv2(img2, desired_encoding='passthrough')
        cv2.imshow("mask_all", img1+img2)
        cv2.waitKey(1)

    def imageCallback(self, rgb):
        cv_image = self.bridge.imgmsg_to_cv2(rgb, desired_encoding='passthrough')
        self.cv_image = cv2.resize(cv_image, (640,480))
        cv2.imshow("mask_all", self.cv_image)
        cv2.waitKey(1)

mask_all_Node()