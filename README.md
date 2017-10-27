# YOLO for Intel/Movidius Neural Compute Stick (NCS)

## News

* Camera App is working.
* YOLOv1 Tiny is working.

## Protobuf Model files

./prototxt/

## Download Pretrained Caffe Models to ./weights/

* YOLO_tiny: https://drive.google.com/file/d/0Bzy9LxvTYIgKNFEzOEdaZ3U0Nms/view?usp=sharing

## Compilation

* Compile .prototxt and corresponding .caffemodel (with the same name) to get NCS graph file. For example: "mvNCCompile prototxt/yolo_tiny_deploy.prototxt -w weights/yolo_tiny_deploy.caffemodel -s 12"
* The compiled binary file "graph" has to be in main folder after this step.

## Single Image Script

* Run "yolo_example.py" to process a single image. For example: "python3 py_examples/yolo_example.py images/dog.jpg" to get detections as below.

![](/images/yolo_dog.png)

## Camera Input Script

* Run "object_detection_app.py" to process a videos from your camera. For example: "python3 py_examples/object_detection_app.py" to get camera detections as below.
* Modify script arguments if needed.
* Press "q" to exit app.

![](/images/camera.png)
