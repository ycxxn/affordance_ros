import cv2
from modeling.deeplab import *
from dataloaders import custom_transforms as tr
from PIL import Image
from torchvision import transforms
from dataloaders.utils import  *
from torchvision.utils import make_grid, save_image
import math 

class build_seg_model:
    def __init__(self, model="mobilenet", class_num=2, ckpt= None):
        self.seg_m = DeepLab(num_classes=class_num,
                    backbone=model,
                    output_stride=16,
                    sync_bn=None,
                    freeze_bn=False)

        ckpt = torch.load(ckpt, map_location='cpu')
        self.seg_m.load_state_dict(ckpt['state_dict'])

        self.composed_transforms = transforms.Compose([
            tr.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            tr.ToTensor()])

        # self.seg_m.cuda()  #up
        self.seg_m.eval()

    def run(self, roi_img):
        roi = cv2.resize(roi_img,(224,224))

        roi = Image.fromarray(roi)
        image = roi.convert("RGB")
        target = roi.convert("L")

        sample = {'image': image, 'label': target}
        tensor_in = self.composed_transforms(sample)['image'].unsqueeze(0)

        # image = image.cuda()
        with torch.no_grad():
            # output = self.seg_m(tensor_in.cuda()) #up
            output = self.seg_m(tensor_in)

        grid_image = make_grid(decode_seg_map_sequence(torch.max(output[:3], 1)[1].detach().cpu().numpy()), 
                            3, normalize=False, range=(0, 255))
    
        grid_image = grid_image.permute(1,2,0)
        grid_image = np.array(grid_image)

        return grid_image

def get_roi(img, boxes, seg_m):
        XYA_inf = [] 
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

            roi_img = cv2.cvtColor(roi_img, cv2.COLOR_BGR2RGB)
            grid_image = seg_m.run(roi_img)
            grid_image = cv2.resize(grid_image,(x2-x1,y2-y1))
            XYA = thresh_mask(grid_image, int(box[6]))
            XYA = local_XYA(XYA,x1,y1)
            XYA_inf.append(XYA)
            mask_all[y1:y2,x1:x2,:] += grid_image
        # for i in XYA_inf:
        #     cv2.circle(mask_all, (i[0][0], i[0][1]), 10, (1, 227, 254), -1)
        #     cv2.circle(mask_all, (i[1][0], i[1][1]), 10, (1, 227, 254), -1)
        return mask_all, XYA_inf

def get_centroid(th):
    # contours, hierarchy = cv2.findContours(th,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    _, contours, hierarchy = cv2.findContours(th,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    areas = [cv2.contourArea(c) for c in contours]
    max_index = np.argmax(areas)
    cnt=contours[max_index]
    M=cv2.moments(cnt)
    cx=int(M['m10']/M['m00'])
    cy=int(M['m01']/M['m00'])
    return [cx,cy]

def thresh_mask(grid_image, classes):
    print(classes)
    if classes == 0 or classes == 1:
        plant_th = grid_image[:,:,0] >= 0.50196078
        handle_th = grid_image[:,:,1] >= 0.50196078

        plant_th = np.array(plant_th, dtype = np.uint8)*255
        handle_th = np.array(handle_th, dtype = np.uint8)*255

        plant_c = get_centroid(plant_th)
        handle_c = get_centroid(handle_th)

        angle = math.atan2(handle_c[1]-plant_c[1], plant_c[0]-handle_c[0])
        # print(plant_c, handle_c, angle*180/math.pi)
        return plant_c, handle_c, angle*180/math.pi
    else:
        cap_th = grid_image[:,:,1] >= 0.50196078
        body_th = grid_image[:,:,0] >= 0.50196078

        cap_th = np.array(cap_th, dtype = np.uint8)*255
        body_th = np.array(body_th, dtype = np.uint8)*255

        cap_c = get_centroid(cap_th)
        body_c = get_centroid(body_th)

        angle = math.atan2(body_c[1]-cap_c[1], cap_c[0]-body_c[0])
        # print(plant_c, handle_c, angle*180/math.pi)
        return cap_c, body_c, angle*180/math.pi

def local_XYA(XYA, x1, y1):
    XYA[0][0] += x1
    XYA[0][1] += y1
    XYA[1][0] += x1
    XYA[1][1] += y1
    return XYA
