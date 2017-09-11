# YOLO for Intel/Movidius Neural Compute Stick (NCS)

## News

* Camera App is working.
* YOLOv1 Tiny is working.

## Protobuf Model files

./prototxt/

## Download Pretrained Caffe Models

* YOLO_tiny: https://drive.google.com/file/d/0Bzy9LxvTYIgKNFEzOEdaZ3U0Nms/view?usp=sharing
* YOLO: https://drive.google.com/file/d/0Bzy9LxvTYIgKMXdqS29HWGNLdGM/view?usp=sharing
* YOLO_small: https://drive.google.com/file/d/0Bzy9LxvTYIgKa3ZHbnZPLUo0eWs/view?usp=sharing

## General Usage

* Compile .prototxt and corresponding .caffemodel (with the same name) to get NCS graph file. For example: "python3 ./mvNCCompile.pyc data/yolo_tiny_deploy.prototxt -s 12"
* Copy graph file to ncapi/networks/YoloTiny/ etc.
* Symlink "./py_examples" into NCS ncapi/py_examples folder

## Single Image Script

* Run "yolo_example.py" to process a single image. For example: "python3 yolo_example.py 1 ../images/dog.jpg" to get detections as below:

![](/images/yolo_dog.png)

## Camera Input Script

* Run "object_detection_app.py" to process a videos from your camera. For example: "python3 object_detection_app.py" to get camera detections as below:

![](/images/camera.png)
