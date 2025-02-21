# Real Time Object Detection Using Machine Learning

#### Introduction:

Welcome to the Git repository for my Master's Project.This Master's Project Git repository in Computer Vision and Machine Learning aims to share my research and development efforts in this exciting area of study. This repository serves as a centralized hub for all the code, documentation, and resources related to my research and development work in this exciting domain. The purpose of this introduction is to provide an overview of the project and guide you through the contents and structure of this repository.

#### Project Overview:

My Master's Project focuses on real-time object detection and number plate recognition. The primary objective is to develop an efficient and accurate system that can detect various objects in real-time video streams and extract number plates from vehicles with high precision. By combining computer vision algorithms and machine learning techniques, this project aims to address the challenges of real-time object detection and number plate recognition in practical scenarios.

### Output :

![image](https://github.com/Shismohammad/Real-Time-Object-Detection-Using-Machine-Learning/blob/master/Images/test.jpg)

![image](https://github.com/Shismohammad/Real-Time-Object-Detection-Using-Machine-Learning/blob/master/Images/test2.jpg)

![image](https://github.com/Shismohammad/Real-Time-Object-Detection-Using-Machine-Learning/blob/master/Images/test3.jpg)

#### Vehicle number plate detection and recognition (Case Study): 

![image](https://github.com/Shismohammad/Real-Time-Object-Detection-Using-Machine-Learning/blob/master/Images/car.jpg)

![image](https://github.com/Shismohammad/Real-Time-Object-Detection-Using-Machine-Learning/blob/master/Images/numberplate.jpg)

![image](https://github.com/Shismohammad/Real-Time-Object-Detection-Using-Machine-Learning/blob/master/Images/car2.jpg)


# Getting Started
### Conda

```bash
#Tensorflow CPU
conda env create -f conda-cpu.yml
conda activate yolov4-cpu

#Tensorflow GPU
conda env create -f conda-gpu.yml
conda activate yolov4-gpu
```

### Pip
```bash
# TensorFlow CPU
pip install -r requirements.txt

# TensorFlow GPU
pip install -r requirements-gpu.txt
```
#### Nvidia Driver (For GPU, if you are not using Conda Environment and haven't set up CUDA yet)
Make sure to use CUDA Toolkit version 10.1 as it is the proper version for the TensorFlow version used in this repository.

## Downloading Official Pre-trained Weights
YOLOv4 comes pre-trained and able to detect 80 classes. For easy demo purposes we used the pre-trained weights.

### Commands:

```bash
python yolo_project.py --weights_location ./weights/yolov4-tiny-416 --model yolov4 --video_location cars_test.mp4

python yolo_project.py --weights_location ./weights/yolov4-tiny-416 --model yolov4 --video_location 0

python yolo_project.py --weights_location ./weights/yolov4-tiny-416 --model yolov3 --video_location ./test/video.mp4
```

```
--output ./detections/results.avi
  --video: path to input video (use 0 for webcam)
    (default: './data/video/video.mp4')
  --output: path to output video (remember to set right codec for given format. e.g. XVID for .avi)
    (default: None)
  --output_format: codec used in VideoWriter when saving video to file
    (default: 'XVID)
  --[no]tiny: yolov4 or yolov4-tiny
    (default: 'false')
  --weights: path to weights file
    (default: './checkpoints/yolov4-416')
 --info: print info on detections
  --model: yolov3 or yolov4
--framework: what framework to use (tf, trt, tflite)
```

#### Counting Objects (total objects or per class)
I have created a custom function within that can be used to count and keep track of the number of objects detected at a given moment within each image or video. It can be used to count total objects found or can count number of objects detected per class.


#### Count Objects Per Class
To count the number of objects for each individual class of your object detector you need to add the custom flag "--count" as well as change one line in the detect.py or detect_video.py script. By default the count_objects function has a parameter called <strong>by_class</strong> that is set to False. If you change this parameter to <strong>True</strong> it will count per class instead.

#### Print Detailed Info About Each Detection (class, confidence, bounding box coordinates)
I have created a custom flag called <strong>INFO</strong> that can be added to any detect.py or detect_video.py commands in order to print detailed information about each detection made by the object detector. To print the detailed information to your command prompt just add the flag "--info" to any of your commands. The information on each detection includes the class, confidence in the detection and the bounding box coordinates of the detection in xmin, ymin, xmax, ymax format.

### References :

It is pretty much hard to implement this from the yolov4 and yolov3 paper alone. I had to reference the official (very hard to understand) and many un-official repos to piece together the complete my roject.

  * [tensorflow-yolov4-tflite](https://github.com/hunglc007/tensorflow-yolov4-tflite)
  * YOLOv4: Optimal Speed and Accuracy of Object Detection [YOLOv4](https://arxiv.org/abs/2004.10934).
  * [darknet](https://github.com/AlexeyAB/darknet)
  * [Yolov3 tensorflow](https://github.com/YunYang1994/tensorflow-yolov3)
  * [Yolov3 tf2](https://github.com/zzh8829/yolov3-tf2)
  * [tensorflow-yolov4-tflite](https://github.com/theAIGuysCode/tensorflow-yolov4-tflite)
