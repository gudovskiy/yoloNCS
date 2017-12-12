#!/usr/bin/env python3
"""
Demo application shows TinyYOLO running on Movidius Neural Compute stick, using RTSP from
surveillance camera as input.

References:
    https://github.com/movidius/ncappzoo/tree/master/caffe/TinyYolo

"""
import sys
import os
import logging

import cv2
from utils.ncs_device import Ncs
from utils.app_utils import FPS, WebcamVideoStream
import numpy as np
from datetime import datetime


logger = logging.getLogger(__name__)

# Tiny Yolo assumes input images are these dimensions.
NETWORK_IMAGE_WIDTH = 448
NETWORK_IMAGE_HEIGHT = 448
WINDOW_NAME = 'TinyYolo (ESC to exit)'

# The 20 classes this network was trained on
TINYYOLO_CLASSES = [
    "aeroplane", "bicycle", "bird", "boat", "bottle",
    "bus", "car", "cat", "chair", "cow",
    "diningtable", "dog", "horse", "motorbike", "person",
    "pottedplant", "sheep", "sofa", "train", "tvmonitor"
]

# A subset of TINYYOLO_CLASSES that I'm interested in
CLASSES_OF_INTEREST = [
    TINYYOLO_CLASSES[1],
    TINYYOLO_CLASSES[6],
    TINYYOLO_CLASSES[7],
    TINYYOLO_CLASSES[12],
    TINYYOLO_CLASSES[14]
]


def boxes_to_pixel_units(box_list, image_width, image_height, grid_size):
    """
    Converts the boxes in box_list to pixel units.  Assumes box_list is the box output
    from the TinyYOLO network and is [grid_size x grid_size x 2 x 4]
    :param box_list:
    :param image_width:
    :param image_height:
    :param grid_size:
    :return: The modified box_list.
    """
    # number of boxes per grid cell
    boxes_per_cell = 2

    # setup some offset values to map boxes to pixels
    # box_offset will be [[ [0, 0], [1, 1], [2, 2], [3, 3], [4, 4], [5, 5], [6, 6]] ...repeated for 7 ]
    box_offset = np.transpose(np.reshape(np.array(
        [np.arange(grid_size)] * (grid_size*2)),
        (boxes_per_cell, grid_size, grid_size)),
        (1, 2, 0)
    )

    # adjust the box center
    box_list[:,:,:,0] += box_offset
    box_list[:,:,:,1] += np.transpose(box_offset,(1,0,2))
    box_list[:,:,:,0:2] = box_list[:,:,:,0:2] / (grid_size * 1.0)

    # adjust the lengths and widths
    box_list[:,:,:,2] = np.multiply(box_list[:,:,:,2],box_list[:,:,:,2])
    box_list[:,:,:,3] = np.multiply(box_list[:,:,:,3],box_list[:,:,:,3])

    # scale the boxes to the image size in pixels
    box_list[:,:,:,0] *= image_width
    box_list[:,:,:,1] *= image_height
    box_list[:,:,:,2] *= image_width
    box_list[:,:,:,3] *= image_height

    return box_list


def iou(box1, box2):
    """
    The intersection-over-union metric determines how close two boxes are to being the same box.
    The closer the boxes are to being the same, the closer the metric will be to 1.0
    box_1 and box_2 are arrays of 4 numbers which are the (x, y) points that define the center
    of the box and the length and width of the box.
    :param box1: nparray coordinates of box1 wrt to its center [x,y,l,w]
    :param box2: nparray coordinates of box2 wrt to its center [x,y,l,w]
    :return: intersection-over-union of the two boxes (between 0.0 and 1.0)
    """
    # Top to bottom of intersecting box
    tb = min(box1[0] + 0.5 * box1[2], box2[0] + 0.5 * box2[2]) - \
        max(box1[0] - 0.5 * box1[2], box2[0] - 0.5 * box2[2])
    # Left to right of intersecting box
    lr = min(box1[1] + 0.5 * box1[3], box2[1] + 0.5 * box2[3]) - \
        max(box1[1] - 0.5 * box1[3], box2[1] - 0.5 * box2[3])

    if tb < 0 or lr < 0:
        # There is no intersecting area.
        intersection = 0
    else:
        # Intersection area is product of tb and lr
        intersection = tb * lr

    # calculate the union area which is the area of each box added
    # and then we need to subtract out the intersection area since
    # it is counted twice (by definition it is in each box)
    union_area = box1[2] * box1[3] + box2[2] * box2[3] - intersection

    return intersection/union_area


def get_duplicate_box_mask(box_list):
    """
    Creates a mask to remove duplicate objects (boxes) and their related probabilities and classifications
    that should be considered the same object.  This is determined by how similar the boxes are
    based on the intersection-over-union metric. box_list is
    :param box_list: A list of boxes (4 floats for centerX, centerY and Length and Width)
    :return:
    """
    # The intersection-over-union threshold to use when determining duplicates.
    # objects/boxes found that are over this threshold will be
    # considered the same object
    max_iou = 0.35

    box_mask = np.ones(len(box_list))

    for i in range(len(box_list)):
        if box_mask[i] == 0:
            continue
        for j in range(i + 1, len(box_list)):
            if iou(box_list[i], box_list[j]) > max_iou:
                box_mask[j] = 0.0

    filter_iou_mask = np.array(box_mask > 0.0, dtype='bool')
    return filter_iou_mask


def filter_objects(inference_result, source_img):
    """
    Interpret the output from a single NCS inference of TinyYolo (GetResult)
    and filter out objects/boxes with low probabilities. Output is the array of floats
    returned from the NCS inference, converted to float32 format.

    :param inference_result: np.ndarray of float16 as returned from NCS device.
    :param source_img: Original RGB image (ndarray) before preprocessing and resizing.
    :return: Returns a list of lists. each of the inner lists represent one found object
        and contains the following 6 values:
           string that is network classification ie 'cat', or 'chair' etc
           float value for box center X pixel location within source_img
           float value for box center Y pixel location within source_img
           float value for box width in pixels within source_img
           float value for box height in pixels within source_img
           float value that is the probability for the network classification.
    """

    # the raw number of floats returned from the inference (GetResult())
    num_inference_results = len(inference_result)

    img_width = source_img.shape[1]
    img_height = source_img.shape[0]

    # only keep boxes with probabilities greater than this
    probability_threshold = 0.07

    num_classifications = len(TINYYOLO_CLASSES)  # should be 20
    grid_size = 7  # the image is a 7x7 grid.  Each box in the grid is 64x64 pixels
    boxes_per_grid_cell = 2  # the number of boxes returned for each grid cell

    # grid_size is 7 (grid is 7x7)
    # num classifications is 20
    # boxes per grid cell is 2
    all_probabilities = np.zeros((grid_size, grid_size, boxes_per_grid_cell, num_classifications))

    # classification_probabilities  contains a probability for each classification for
    # each 64x64 pixel square of the grid.  The source image contains
    # 7x7 of these 64x64 pixel squares and there are 20 possible classifications
    classification_probabilities = \
        np.reshape(inference_result[0:980], (grid_size, grid_size, num_classifications))
    num_of_class_probs = len(classification_probabilities)

    # The probability scale factor for each box
    box_prob_scale_factor = np.reshape(inference_result[980:1078], (grid_size, grid_size, boxes_per_grid_cell))

    # get the boxes from the results and adjust to be pixel units
    all_boxes = np.reshape(inference_result[1078:], (grid_size, grid_size, boxes_per_grid_cell, 4))
    boxes_to_pixel_units(all_boxes, img_width, img_height, grid_size)

    # adjust the probabilities with the scaling factor
    for box_index in range(boxes_per_grid_cell): # loop over boxes
        for class_index in range(num_classifications): # loop over classifications
            all_probabilities[:,:,box_index,class_index] = np.multiply(classification_probabilities[:,:,class_index],box_prob_scale_factor[:,:,box_index])

    probability_threshold_mask = np.array(all_probabilities >= probability_threshold, dtype='bool')
    box_threshold_mask = np.nonzero(probability_threshold_mask)
    boxes_above_threshold = all_boxes[box_threshold_mask[0],box_threshold_mask[1],box_threshold_mask[2]]
    classifications_for_boxes_above = np.argmax(all_probabilities, axis=3)[box_threshold_mask[0],box_threshold_mask[1],box_threshold_mask[2]]
    probabilities_above_threshold = all_probabilities[probability_threshold_mask]

    # sort the boxes from highest probability to lowest and then
    # sort the probabilities and classifications to match
    argsort = np.array(np.argsort(probabilities_above_threshold))[::-1]
    boxes_above_threshold = boxes_above_threshold[argsort]
    classifications_for_boxes_above = classifications_for_boxes_above[argsort]
    probabilities_above_threshold = probabilities_above_threshold[argsort]

    # get mask for boxes that seem to be the same object
    duplicate_box_mask = get_duplicate_box_mask(boxes_above_threshold)

    # update the boxes, probabilities and classifications removing duplicates.
    boxes_above_threshold = boxes_above_threshold[duplicate_box_mask]
    classifications_for_boxes_above = classifications_for_boxes_above[duplicate_box_mask]
    probabilities_above_threshold = probabilities_above_threshold[duplicate_box_mask]

    classes_boxes_and_probs = []
    for i in range(len(boxes_above_threshold)):
        object_name = TINYYOLO_CLASSES[classifications_for_boxes_above[i]]
        if object_name in CLASSES_OF_INTEREST:  # Uncomment this line to filter out uninteresting objects
            classes_boxes_and_probs.append([
                object_name,
                boxes_above_threshold[i][0],
                boxes_above_threshold[i][1],
                boxes_above_threshold[i][2],
                boxes_above_threshold[i][3],
                probabilities_above_threshold[i]
            ])

    return classes_boxes_and_probs


def display_objects_in_gui(source_image, filtered_objects):
    """
    Displays a gui window with an image that contains boxes and labels for found objects.
    :param source_image: The original image that was given to NCS, before preprocessing.
    :param filtered_objects: a list of lists. each of the inner lists represent one found object
        and contains the following 6 values:
           string that is network classification ie 'cat', or 'chair' etc
           float value for box center X pixel location within source_img
           float value for box center Y pixel location within source_img
           float value for box width in pixels within source_img
           float value for box height in pixels within source_img
           float value that is the probability for the network classification.
    :return: A stop_flag where True=user abort, False=continue displaying.
    """
    # copy image so we can draw on it.
    display_image = source_image.copy()
    source_image_width = source_image.shape[1]
    source_image_height = source_image.shape[0]

    # loop through each box and draw it on the image along with a classification label
    for obj_index in range(len(filtered_objects)):
        object_name = filtered_objects[obj_index][0]
        center_x = int(filtered_objects[obj_index][1])
        center_y = int(filtered_objects[obj_index][2])
        half_width = int(filtered_objects[obj_index][3]) // 2
        half_height = int(filtered_objects[obj_index][4]) // 2
        confidence = filtered_objects[obj_index][5]

        # calculate box (left, top) and (right, bottom) coordinates
        box_left = max(center_x - half_width, 0)
        box_top = max(center_y - half_height, 0)
        box_right = min(center_x + half_width, source_image_width)
        box_bottom = min(center_y + half_height, source_image_height)

        logger.debug('Object[%d of %d]=%s   [l,t,r,b]=[%d,%d,%d,%d]   Confidence=%0.2f' %
            (obj_index, len(filtered_objects), object_name, box_left, box_top, box_right, box_bottom, confidence))

        # draw the rectangle on the image.  This is hopefully around the object
        box_color = (0, 255, 0)  # green box
        box_thickness = 3
        cv2.rectangle(display_image, (box_left, box_top), (box_right, box_bottom), box_color, box_thickness)

        # draw the classification label string just above and to the left of the rectangle
        label_background_color = (70, 120, 70)  # greyish green background for text
        label_text_color = (255, 255, 255)  # white text
        cv2.rectangle(display_image, (box_left, box_top - 20), (box_right, box_top), label_background_color, -1)
        label_text = object_name + ' : %.2f' % confidence
        cv2.putText(
            display_image,
            label_text,
            (box_left + 5, box_top - 7),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            label_text_color,
            1)

    cv2.imshow(WINDOW_NAME, display_image)
    key = cv2.waitKey(1)
    if key & 0xFF == 27:
        return True  # user exit
    return False


def img_preprocess(img):
    """
    This performs preprocessing on a raw RGB image to prepare it for input into the NCS.
    YOLO wants input images to have dimensions of 448x448 pixels.
    NCS wants input tensors to be an array of float16 ranging from 0-1.000
    :param img:  A ndarray object of shape [x, y, 3] which contains BGR values (0-255)
    :return: A resized ndarray of shape [448, 448, 3] with RGB float16 values in range 0.0-1.0
    """
    dim = (NETWORK_IMAGE_WIDTH, NETWORK_IMAGE_HEIGHT)
    tensor_image = cv2.resize(src=img, dsize=dim, interpolation=cv2.INTER_LINEAR)
    tensor_image = cv2.cvtColor(src=tensor_image, code=cv2.COLOR_BGR2RGB)
    tensor_image = np.divide(tensor_image, 255.0)
    # tensor_image = resize(img.copy() / 255.0, dim, 1)
    # tensor_image = tensor_image[:, :, (2, 1, 0)]  # Does this change from RGB to BGR?
    # print('NEW shape:',im.shape)
    # print(img[0,0,:],im[0,0,:])
    return tensor_image.astype(np.float16)


def main(argv):
    app_start_time = datetime.now()
    logging.basicConfig(
        stream=sys.stdout,
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
        sys.exit('Usage: python3 ' + __file__ + ' <rtsp_uri>')
    rtsp_uri = argv[0]
    logger.info('Using stream {}'.format(rtsp_uri))

    # by changing the NETWORK_DIR environment variable, you can select between YoloTiny and YoloSmall
    net_dir = os.getenv('NETWORK_DIR')
    if net_dir is None:
        sys.exit('Missing NETWORK_DIR env path to YOLO [Tiny|Small] graph file')
    logger.info('Using network {}'.format(net_dir))

    # set up a window to show results
    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    #cv2.resizeWindow(WINDOW_NAME, 960, 540)
    cv2.resizeWindow(WINDOW_NAME, 1920, 1080)

    # set up the NCS device
    Ncs.set_log_level(2)
    gp = os.path.join(net_dir, 'graph')

    with WebcamVideoStream(src=rtsp_uri) as stream:
        with Ncs(graph_path=gp) as ncs:
            stop_flag = False
            while stop_flag is not True:
                # grab frame from stream
                img = stream.read()
                tensor = img_preprocess(img)

                # run a single inference on the tensor
                out, _ = ncs.infer(tensor)

                # filter out all the objects/boxes that don't meet thresholds.
                # This will return fc27 instead of fc12 for yolo_small.
                results = filter_objects(out.astype(np.float32), img)

                # display the filtered objects/boxes in a GUI window
                stop_flag = display_objects_in_gui(img, results)

    cv2.destroyAllWindows()

    uptime = datetime.now() - app_start_time
    logger.info(
        '\n'
        '-------------------------------------------------------------------\n' 
        '   Stopped {0}\n'
        '   Uptime was {1}\n'
        '-------------------------------------------------------------------\n'
        .format(__file__, str(uptime)))
    logging.shutdown()

if __name__ == '__main__':
    main(sys.argv[1:])
