# YOLO for Intel/Movidius Neural Compute Stick (NCS)

## News

* YOLOv1 Tiny is working.
* YOLOv1 Small is not working.
* YOLOv1 Full is not checked.

## Protobuf Model files

./prototxt/

## Download Pretrained Caffe Models

* YOLO: https://drive.google.com/file/d/0Bzy9LxvTYIgKMXdqS29HWGNLdGM/view?usp=sharing
* YOLO_small: https://drive.google.com/file/d/0Bzy9LxvTYIgKa3ZHbnZPLUo0eWs/view?usp=sharing
* YOLO_tiny: https://drive.google.com/file/d/0Bzy9LxvTYIgKNFEzOEdaZ3U0Nms/view?usp=sharing

## Usage

* Compile .prototxt and corresponding .caffemodel (with the same name) to get NCS graph file. For example: "python3 ./mvNCCompile.pyc data/yolo_tiny_deploy.prototxt -s 12"
* Copy graph file to ncapi/networks/YoloTiny/ etc.
* Copy "./py_examples/yolo_example.py" into NCS ncapi/py_examples folder
* Run "yolo_example.py" to process a single image. For example: "python3 yolo_example.py 1 ../images/dog.jpg"
* You should get an image below:

![](/images/yolo_dog.png)
