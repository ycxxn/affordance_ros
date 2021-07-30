import argparse
import numpy as np 
import cv2
import rospy
from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import Image
from modeling.deeplab import *
# from dataloaders import custom_transforms as tr
from PIL import Image
# from torchvision import transforms
# from dataloaders.utils import  *
# from torchvision.utils import make_grid, save_image


class Node:
    def __init__(self):
        rospy.init_node("affordace_node")
        self.bridge = CvBridge()
        # self.init_seg()
        rospy.Subscriber("/camera/color/image_raw", Image, self.imageCallback)

        rospy.spin()

    def imageCallback(self, rgb):
        cv_image = self.bridge.imgmsg_to_cv2(rgb, "bgr8")
        self.cv_image = cv2.resize(cv_image, (640,480))
        cv2.imshow("1",self.cv_image)
        cv2.waitKey(1)

if __name__ == '__main__':
    Node()