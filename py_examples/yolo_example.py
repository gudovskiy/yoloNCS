#!/usr/bin/env python3
"""
How to compile the YoloTiny graph:
piero@piero-ubuntu:~/movidius/bin$ python3 mvNCCompile.pyc /home/piero/movidius/ncapi/networks/YoloTiny/yolo_tiny_deploy.prototxt -w /home/piero/movidius/ncapi/networks/YoloTiny/yolo_tiny.caffemodel -s 12 -o /home/piero/movidius/ncapi/networks/YoloTiny/yolo_tiny_graph

TODO
Capture a single RTSP from from a camera and count persons.
ffmpeg -y -i rtsp://admin:admin@10.0.90.71/ufirststream -vframes 1 do.jpg

"""
import sys
import os
import logging

import cv2
from utils.ncs_device import Ncs
import numpy as np
from datetime import datetime
from skimage.transform import resize


logger = logging.getLogger(__name__)


def interpret_output(output, img_width, img_height):
    classes = [
        "aeroplane", "bicycle", "bird", "boat", "bottle",
        "bus", "car", "cat", "chair", "cow",
        "diningtable", "dog", "horse", "motorbike", "person",
        "pottedplant", "sheep", "sofa", "train", "tvmonitor"
    ]
    w_img = img_width
    h_img = img_height

    # some interpretation constants
    threshold = 0.2  # any probability over .20 is considered a detection
    iou_threshold = 0.5
    num_class = len(classes)  # 20
    num_box = 2
    grid_size = 7
    class_grid = (grid_size, grid_size, num_class)

    # extract the class probabilities. Each square of the grid contains a sequence of
    # float32 probabilities for each enumerated object class type.
    probs = np.zeros((grid_size, grid_size, 2, num_class))
    class_probs = (np.reshape(output[0:980], class_grid))  # .copy()
    # print(class_probs)

    scales = (np.reshape(output[980:1078], (7, 7, 2)))  # .copy()
    # print(scales)
    boxes = (np.reshape(output[1078:], (7, 7, 2, 4)))  # .copy()
    offset = np.transpose(np.reshape(np.array([np.arange(7)] * 14), (2, 7, 7)), (1, 2, 0))
    # boxes.setflags(write=1)
    boxes[:, :, :, 0] += offset
    boxes[:, :, :, 1] += np.transpose(offset, (1, 0, 2))
    boxes[:, :, :, 0:2] = boxes[:, :, :, 0:2] / 7.0
    boxes[:, :, :, 2] = np.multiply(boxes[:, :, :, 2], boxes[:, :, :, 2])
    boxes[:, :, :, 3] = np.multiply(boxes[:, :, :, 3], boxes[:, :, :, 3])

    boxes[:, :, :, 0] *= w_img
    boxes[:, :, :, 1] *= h_img
    boxes[:, :, :, 2] *= w_img
    boxes[:, :, :, 3] *= h_img

    for i in range(2):
        for j in range(num_class):
            probs[:, :, i, j] = np.multiply(class_probs[:, :, j], scales[:, :, i])
        # print (probs)
    filter_mat_probs = np.array(probs >= threshold, dtype='bool')
    filter_mat_boxes = np.nonzero(filter_mat_probs)
    boxes_filtered = boxes[filter_mat_boxes[0], filter_mat_boxes[1], filter_mat_boxes[2]]
    probs_filtered = probs[filter_mat_probs]
    classes_num_filtered = np.argmax(probs, axis=3)[filter_mat_boxes[0], filter_mat_boxes[1], filter_mat_boxes[2]]

    argsort = np.array(np.argsort(probs_filtered))[::-1]
    boxes_filtered = boxes_filtered[argsort]
    probs_filtered = probs_filtered[argsort]
    classes_num_filtered = classes_num_filtered[argsort]

    for i in range(len(boxes_filtered)):
        if probs_filtered[i] == 0:
            continue
        for j in range(i + 1, len(boxes_filtered)):
            if iou(boxes_filtered[i], boxes_filtered[j]) > iou_threshold:
                probs_filtered[j] = 0.0

    filter_iou = np.array(probs_filtered > 0.0, dtype='bool')
    boxes_filtered = boxes_filtered[filter_iou]
    probs_filtered = probs_filtered[filter_iou]
    classes_num_filtered = classes_num_filtered[filter_iou]

    result = []
    for i in range(len(boxes_filtered)):
        result.append(
            [classes[classes_num_filtered[i]], boxes_filtered[i][0], boxes_filtered[i][1], boxes_filtered[i][2],
             boxes_filtered[i][3], probs_filtered[i]])

    return result


def iou(box1, box2):
    tb = min(box1[0] + 0.5 * box1[2], box2[0] + 0.5 * box2[2]) - \
         max(box1[0] - 0.5 * box1[2], box2[0] - 0.5 * box2[2])
    lr = min(box1[1] + 0.5 * box1[3], box2[1] + 0.5 * box2[3]) - \
         max(box1[1] - 0.5 * box1[3], box2[1] - 0.5 * box2[3])
    if tb < 0 or lr < 0:
        intersection = 0
    else:
        intersection = tb * lr
    return intersection / (box1[2] * box1[3] + box2[2] * box2[3] - intersection)


def show_results(img, results, img_width, img_height):
    img_cp = img.copy()
    disp_console = True
    imshow = True
    #	if self.filewrite_txt :
    #		ftxt = open(self.tofile_txt,'w')
    for i in range(len(results)):
        x = int(results[i][1])
        y = int(results[i][2])
        w = int(results[i][3]) // 2
        h = int(results[i][4]) // 2
        if disp_console:
            print('    class : ' + results[i][0] + ' , [x,y,w,h]=[' + str(x) + ',' + str(y) + ',' + str(
                int(results[i][3])) + ',' + str(int(results[i][4])) + '], Confidence = ' + str(results[i][5]))
        xmin = x - w
        xmax = x + w
        ymin = y - h
        ymax = y + h
        if xmin < 0:
            xmin = 0
        if ymin < 0:
            ymin = 0
        if xmax > img_width:
            xmax = img_width
        if ymax > img_height:
            ymax = img_height
        if imshow:
            cv2.rectangle(img_cp, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
            # print ((xmin, ymin, xmax, ymax))
            cv2.rectangle(img_cp, (xmin, ymin - 20), (xmax, ymin), (125, 125, 125), -1)
            cv2.putText(img_cp, results[i][0] + ' : %.2f' % results[i][5], (xmin + 5, ymin - 7),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
    if imshow:
        cv2.imshow('YOLO detection', img_cp)
        cv2.waitKey(1000)


def img_preprocess(img):
    """
    This performs preprocessing on a raw RGB image to prepare it for input into the NCS.
    YOLO wants input images to have dimensions of 448x448 pixels.
    NCS wants input tensors to be an array of float16 ranging from 0-1.000
    :param img:  A ndarray object of shape [x, y, 3] which contains BGR values (0-255)
    :return: A resized ndarray of shape [448, 448, 3] with RGB float16 values in range 0.0-1.0
    """
    dim = (448, 448)
    tensor_image = cv2.resize(src=img, dsize=dim)
    tensor_image = cv2.cvtColor(src=tensor_image, code=cv2.COLOR_BGR2RGB)
    tensor_image = tensor_image / 255.0
    # tensor_image = resize(img.copy() / 255.0, dim, 1)
    # tensor_image = tensor_image[:, :, (2, 1, 0)]  # Does this change from RGB to BGR?
    # print('NEW shape:',im.shape)
    # print(img[0,0,:],im[0,0,:])
    return tensor_image.astype(np.float16)


def main(argv):
    app_start_time = datetime.now()
    logging.basicConfig(
        format='%(asctime)s.%(msecs)03d %(name)-12s %(levelname)-8s [%(threadName)-12s] %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    logger.setLevel(logging.DEBUG)

    logger.info(
        '\n'
        '-------------------------------------------------------------------\n'
        '    Running {0}\n'
        '    Started on {1}\n'
        '-------------------------------------------------------------------\n'
        .format(__file__, app_start_time.isoformat())
    )

    if len(argv) == 0:
        sys.exit('Usage: python3 ' + __file__ + ' <image_file>')
    img_file = argv[0]
    logger.info('Using image {}'.format(img_file))

    # by changing the NETWORK_DIR environment variable, you can select between YoloTiny and YoloSmall
    net_dir = os.getenv('NETWORK_DIR')
    if net_dir is None:
        sys.exit('Missing NETWORK_DIR env path to YOLO [Tiny|Small] graph file')
    logger.info('Using network {}'.format(net_dir))

    Ncs.set_log_level(2)
    gp = os.path.join(net_dir, 'graph')
    with Ncs(graph_path=gp) as ncs:
        # convert RGB image data into tensor of float16
        img = cv2.imread(img_file)
        tensor = img_preprocess(img)

        # run a single inference on the tensor
        out, userobj = ncs.infer(tensor)

        results = interpret_output(
            out.astype(np.float32),
            img.shape[1],
            img.shape[0]
        )  # fc27 instead of fc12 for yolo_small

        show_results(img, results, img.shape[1], img.shape[0])
        cv2.waitKey(10000)

    uptime = datetime.now() - app_start_time
    logger.info(
        '\n'
        '-------------------------------------------------------------------\n' 
        '   Stopped {0}\n'
        '   Uptime was {1}\n'
        '-------------------------------------------------------------------\n'
        .format(__file__, str(uptime)))
    logging.shutdown()

if __name__ == "__main__":
    main(sys.argv[1:])
