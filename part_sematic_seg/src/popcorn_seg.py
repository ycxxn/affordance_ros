#!/usr/bin/env python3  

import numpy as np 
import cv2
import rospy
import rospkg
import sys
sys.path.insert(0, '/opt/installer/open_cv/cv_bridge/lib/python3/dist-packages/')

from cv_bridge import CvBridge, CvBridgeError

from sensor_msgs.msg import Image as msg_Image
from part_detection.msg import yolo_bboxes
from part_sematic_seg.msg import XYA
from part_sematic_seg.msg import XYAs

from seg_model import build_seg_model, get_roi

class popcorn_node:
    def __init__(self):
        rospy.init_node('popcorn_node')
        self.bridge = CvBridge()
        self.img_width = 640
        self.img_height = 480        
        # self.seg_m = build_seg_model(model="mobilenet", class_num=3, ckpt="./weights/seg_mobilenet_popcorn.pth")
        pkg_path = rospkg.RosPack().get_path('part_sematic_seg')
        self.seg_m = build_seg_model(model="mobilenet", class_num=3, ckpt=pkg_path +"/weights/seg_mobilenet_popcorn.pth")
        rospy.Subscriber("/camera/color/image_raw", msg_Image, self.imageCallback)
        rospy.Subscriber("popcorn_pub", yolo_bboxes, self.popcorn_callback)

        self.pub_xya = rospy.Publisher('popcorn_xya', XYAs, queue_size=10)
        self.image_pub = rospy.Publisher("popcorn_seg_img", msg_Image)
        rospy.spin()

    def popcorn_callback(self, data_bboxes):
        print('popcorn_bboxes', data_bboxes)
        if len(data_bboxes.bboxes) == 0:
            self.popcorn_bboxes = []
        else:
            self.popcorn_bboxes = data_bboxes.bboxes
    
    def imageCallback(self, rgb):
        cv_image = self.bridge.imgmsg_to_cv2(rgb, "bgr8")
        self.cv_image = cv2.resize(cv_image, (self.img_width, self.img_height))
        self.seg()

    def seg(self):
        mask_all, XYA_info = get_roi(self.cv_image, self.popcorn_bboxes, self.seg_m)

        XYA_list = XYAs()
        for i in XYA_info:
            XYA_msg = XYA()
            XYA_msg.centroid1_x = i[0][0]
            XYA_msg.centroid1_y = i[0][1]
            XYA_msg.centroid2_x = i[1][0]
            XYA_msg.centroid2_y = i[1][1]
            XYA_msg.angle = i[2]

            XYA_list.xyas.append(XYA_msg)

            cv2.circle(mask_all, (i[0][0], i[0][1]), 10, (1, 227, 254), -1)
            cv2.circle(mask_all, (i[1][0], i[1][1]), 10, (1, 227, 254), -1)

        rospy.loginfo(XYA_list)
        self.pub_xya.publish(XYA_list)      
        self.image_pub.publish(self.bridge.cv2_to_imgmsg(mask_all, encoding="passthrough"))

        # cv2.imshow("popcorn_seg", mask_all)
        # cv2.waitKey(1)
       
if __name__ == '__main__':
    popcorn_node()
