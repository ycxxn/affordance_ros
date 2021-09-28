## download weighting for YOLOv4, DeepLab V3+
[YOLOv4 weight link](https://drive.google.com/drive/folders/1dlr5jqBtIn0p6nySPJwbV-puL-CtW10m?usp=sharing)
put files in folder workspace/src/affordance_ros/part_detection/weights/20210806/

[DeepLab V3+ weight link](https://drive.google.com/drive/folders/17LXPeecffG_9KakoZ3LiVsIZg1lx6moD?usp=sharing)
put files in folder workspace/src/affordance_ros/part_sematic_seg/weights/

```
#YOLOv4
chomod +x demo_d435i.py

. devel/setup.bash
roslaunch part_detection part_detection.launch

#DeepLab V3+
chomod +x juice_seg.py
chomod +x popcorn_seg.py
chomod +x show_seg_all.py

. devel/setup.bash
rosrun part_sematic_seg juice_seg.py
rosrun part_sematic_seg popcorn_seg.py
rosrun part_sematic_seg show_seg_all.py
```

## Dependencies
* Install [realsense-ros](https://github.com/IntelRealSense/realsense-ros)
* Install [cv_bridge for Python3]()

## Software environment
* Ubuntu 18.04
* ROS Melodic
* Python 3.6.9
* opencv 4.5.1 cv2 (?)
* cv_bridge (python3 待測試) (?)
* Install pcl (?)
* Install cv_bridge(?)

### 1. 部件分割 affordance_ros
   + (1) `part_detection: YOLOv4, YOLOv4-tiny` 
    - popcorn_f, popcorn_b, juice  
    <img src="readme_img/part_detection.png" alt="drawing" width="300"/>  
    part_detection_result
  
   + (2) `part_sematic_seg: DeepLab V3+`
    - plant, handle, cap, body  
    <img src="readme_img/part_sematic_seg(juice).png" alt="drawing" width="298"/>
    <img src="readme_img/part_sematic_seg(popcorn).png" alt="drawing" width="300"/>  
    part_sematic_seg(juice), part_sematic_seg(popcorn)  
      
     <img src="readme_img/part_sematic_seg(juice,popcorn).png" alt="drawing" width="300"/>  
    part_sematic_seg(juice,popcorn)