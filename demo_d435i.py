# -*- coding: utf-8 -*-
import argparse
from matplotlib.pyplot import jet
import numpy as np 
import cv2
import rospy
from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import Image as msg_Image
#detection
from torch._C import dtype, unify_type_list
from tool.utils import *
from tool.torch_utils import *
from tool.darknet2pytorch import Darknet
# segementation
from modeling.deeplab import *
from dataloaders import custom_transforms as tr
from PIL import Image
from torchvision import transforms
from dataloaders.utils import  *
from torchvision.utils import make_grid, save_image
"""hyper parameters"""
use_cuda = True

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

def get_centroid(th):
    contours, hierarchy = cv2.findContours(th,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    areas = [cv2.contourArea(c) for c in contours]
    max_index = np.argmax(areas)
    cnt=contours[max_index]
    M=cv2.moments(cnt)
    cx=int(M['m10']/M['m00'])
    cy=int(M['m01']/M['m00'])
    return [cx,cy]

def thresh_mask(grid_image):
    plant_th = grid_image[:,:,0] >= 0.50196078
    handle_th = grid_image[:,:,1] >= 0.50196078

    plant_th = np.array(plant_th, dtype = np.uint8)*255
    handle_th = np.array(handle_th, dtype = np.uint8)*255

    plant_c = get_centroid(plant_th)
    handle_c = get_centroid(handle_th)

    angle = math.atan2(handle_c[1]-plant_c[1], plant_c[0]-handle_c[0])
    # print(angle*180/math.pi)
    # cv2.circle(grid_image, plant_c, 10, (1, 227, 254), -1)
    # cv2.circle(grid_image, handle_c, 10, (1, 227, 254), -1)
    # cv2.imshow("plant_th", plant_th)
    # cv2.imshow("handle_th", handle_th)
    return plant_c, handle_c, angle*180/math.pi

class Node:
    def __init__(self, cfg, w):
        rospy.init_node("affordace_node")
        self.bridge = CvBridge()
        self.init_yolo(cfg, w)
        self.init_seg()
        rospy.Subscriber("/camera/color/image_raw", msg_Image, self.imageCallback)
        rospy.Subscriber("/camera/depth/image_rect_raw", msg_Image, self.imageCallback2)
        
        rospy.spin()

    def init_yolo(self, cfg, w):
        m = Darknet(cfg)
        m.print_network()
        m.load_weights(w)
        if use_cuda:
            m.cuda()
        self.m = m

        namesfile = 'data/obj.names'
        self.class_names = load_class_names(namesfile)

    def init_seg(self):
        self.seg_m = DeepLab(num_classes=3,
                    backbone="mobilenet",
                    output_stride=16,
                    sync_bn=None,
                    freeze_bn=False)

        ckpt = torch.load("./weights/deeplab-mobilenet.pth", map_location='cpu')
        self.seg_m.load_state_dict(ckpt['state_dict'])

        self.composed_transforms = transforms.Compose([
            tr.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            tr.ToTensor()])

        self.seg_m.eval()
        # cuda_use = not True and torch.cuda.is_available()


    def imageCallback(self, rgb):
        cv_image = self.bridge.imgmsg_to_cv2(rgb, "bgr8")
        self.cv_image = cv2.resize(cv_image, (640,480))
    
    def imageCallback2(self, depth):
        depth = self.bridge.imgmsg_to_cv2(depth, "32FC1")
        self.depth = cv2.resize(depth, (640,480))

        self.det()

    def get_roi(self, img, boxes):
        width = img.shape[1]
        height = img.shape[0]
        mask_all = np.zeros((height, width, 3))
        for i in range(len(boxes)):
            box = boxes[i]
            # if box[4]>thresh:
            x1 = int(box[0] * width)
            y1 = int(box[1] * height)
            x2 = int(box[2] * width)
            y2 = int(box[3] * height)
            x1,y1,x2,y2 = np.clip([x1,y1,x2,y2],0,9999)
            roi_img = img[y1:y2,x1:x2,:]

            grid_image = self.affordance(roi_img)
            grid_image = cv2.resize(grid_image,(x2-x1,y2-y1))
            mask_all[y1:y2,x1:x2,:] += grid_image
        return mask_all
            # cv2.imshow("roi", mask_all)

    def affordance(self, roi_img):
        roi = cv2.resize(roi_img,(224,224))

        roi = Image.fromarray(roi)
        image = roi.convert("RGB")
        target = roi.convert("L")

        sample = {'image': image, 'label': target}
        tensor_in = self.composed_transforms(sample)['image'].unsqueeze(0)

        # image = image.cuda()
        with torch.no_grad():
            output = self.seg_m(tensor_in)

        grid_image = make_grid(decode_seg_map_sequence(torch.max(output[:3], 1)[1].detach().cpu().numpy()), 
                            3, normalize=False, range=(0, 255))
    
        grid_image = grid_image.permute(1,2,0)
        grid_image = np.array(grid_image)
        # cv2.imshow("grid_image", grid_image)
        return grid_image


    def det(self):
        sized = cv2.resize(self.cv_image, (self.m.width, self.m.height))
        sized = cv2.cvtColor(sized, cv2.COLOR_BGR2RGB)

        boxes = do_detect(self.m, sized, 0.20, 0.6, use_cuda)

        mask_all = self.get_roi(self.cv_image, boxes[0])
        result_img = plot_boxes_cv2(self.cv_image, boxes[0], savename=None, class_names=self.class_names)

        plant_c, handle_c, angle = thresh_mask(mask_all)

        plant_depth = self.depth[plant_c[1],plant_c[0]]
        handle_depth = self.depth[handle_c[1],handle_c[0]]

        rospy.loginfo("plant_centrid: {}, depth:{}".format(plant_c, plant_depth))
        rospy.loginfo("handle_centrid:{}, depth:{}".format(handle_c, handle_depth))
        rospy.loginfo("angle: {}".format(angle))

        cv2.circle(mask_all, plant_c, 10, (1, 227, 254), -1)
        cv2.circle(mask_all, handle_c, 10, (1, 227, 254), -1)

        cv2.imshow("mask_all", mask_all)
        cv2.imshow("result_img", result_img)
    
        cv2.waitKey(1)

    

if __name__ == '__main__':
    args = get_args()
    Node(args.cfgfile, args.weightfile)