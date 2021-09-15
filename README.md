download weighting for YOLOv4
https://drive.google.com/drive/folders/1dlr5jqBtIn0p6nySPJwbV-puL-CtW10m?usp=sharing
put files in folder
<workspace>/src/affordance_ros/part_detection/weights/20210806/


#YOLOv4
chomod +x demo_d435i.py

. devel/setup.bash
roslaunch part_detection part_detection.launch

#DeepLab v3 Plus
chomod +x juice_seg.py
chomod +x popcorn_seg.py
chomod +x show_seg_all.py

. devel/setup.bash
rosrun part_sematic_seg juice_seg.py
rosrun part_sematic_seg popcorn_seg.py
rosrun part_sematic_seg show_seg_all.py

