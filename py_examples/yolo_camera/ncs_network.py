import numpy
import os
from skimage.transform import resize

class NcsResult:
  def __init__(self):
    self.top = []

  def __str__(self):
    text = ""
    for cat in self.top:
      text += cat[0] + ' (' + '{0:.2f}'.format(cat[1]*100) + '%) '
    return text

  def add(self,category,probability):
    self.top.append((category,probability))

class NcsNetwork:
  """ Class representing a graph
  """

  def __init__(self,directory):
    self.network_dir = directory
    #Load preprocessing data
    #with open(os.path.join(self.network_dir,'stat.txt'), 'r') as f:
    #    self.mean = f.readline().split()
    #    self.std = f.readline().split()
    #    for i in range(3):
    #            self.mean[i] = 255 * float(self.mean[i])
    #            self.std[i] = 1 / (255 * float(self.std[i]))
    #Load categories
    #self.categories = []
    #with open(os.path.join(self.network_dir,'categories.txt'), 'r') as f:
    #    for line in f:
    #            cat = line.split('\n')[0]
    #            if cat != 'classes':
    #                    self.categories.append(cat)
    #    f.close()
    #    # print('Number of categories:', len(self.categories))
    #Load image size
    #with open(os.path.join(self.network_dir,'inputsize.txt'), 'r') as f:
    #   self.reqsize = int(f.readline().split('\n')[0])
    self.reqsize = 448
    self.classes = ["aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow", "diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"]

  def rawinputformat(self):
    """ returns a tuple of (width, height, format)
    describing the input format before preprocessing
    """
    return (self.reqsize, self.reqsize,"RGB")

  def get_graph_binary(self):
    """ returns the graph binary
    """
    with open(os.path.join(self.network_dir,"graph"), mode='rb') as file:
      graph = file.read()
    return graph

  def preprocess(self, data):
    """ preprocess a video frame
    input - in the format specified by rawinputformat() method
    output - in the format required by the graph
    """
    (w,h,f) = self.rawinputformat()
    dt = numpy.dtype(numpy.uint8)
    nb = numpy.frombuffer(data,dt,-1,0)
    actual_stream_width = (w&1)+w # hack, rather get this from the app sink
    if(actual_stream_width != self.reqsize):
        nb = nb.reshape(h,actual_stream_width,3)
        nb = nb[0:h,0:w,0:3] # crop to network input size
    else:
        nb = nb.reshape((actual_stream_width,actual_stream_width,3))
        img = nb.astype('float32')
    #Preprocess image
    #for i in range(3):
    #    img[:,:,i] = (img[:,:,i] - self.mean[i]) * self.std[i]
    #img = resize(img/255.0,(w,h),1)
    img = img/255.0
    print(img.shape)
    #print(img[0,0,:])
    return img.astype(numpy.float16)

  def postprocess(self,graph_output):
    """ postprocess an inference result
    graph_output - in the format produced by the graph
    return value - in a human readable format
    """
    (w,h,f) = self.rawinputformat()
    #print(w,h)
    #print(graph_output.shape)
    results = self.interpret_output(graph_output.astype(numpy.float32), w, h)
    print(results)
    #order = graph_output.argsort()
    #last = len(self.categories)-1
    #res = NcsResult();
    #print(len(results))
    #for i in range(0,len(results)):
    #    res = results[i]
        #res.add( self.categories[order[last-i]], ( graph_output[order[last-i]] ) )
    return res

    # YOLO stuff
  def interpret_output(self,output,img_width,img_height):
    w_img = img_width
    h_img = img_height
    #print ((w_img, h_img))
    threshold = 0.2
    iou_threshold = 0.5
    num_class = 20
    num_box = 2
    grid_size = 7
    probs = numpy.zeros((7,7,2,20))
    class_probs = (numpy.reshape(output[0:980],(7,7,20)))#.copy()
    #print(class_probs)
    scales = (numpy.reshape(output[980:1078],(7,7,2)))#.copy()
    #print(scales)
    boxes = (numpy.reshape(output[1078:],(7,7,2,4)))#.copy()
    offset = numpy.transpose(numpy.reshape(numpy.array([numpy.arange(7)]*14),(2,7,7)),(1,2,0))
    #boxes.setflags(write=1)
    boxes[:,:,:,0] += offset
    boxes[:,:,:,1] += numpy.transpose(offset,(1,0,2))
    boxes[:,:,:,0:2] = boxes[:,:,:,0:2] / 7.0
    boxes[:,:,:,2] = numpy.multiply(boxes[:,:,:,2],boxes[:,:,:,2])
    boxes[:,:,:,3] = numpy.multiply(boxes[:,:,:,3],boxes[:,:,:,3])

    boxes[:,:,:,0] *= w_img
    boxes[:,:,:,1] *= h_img
    boxes[:,:,:,2] *= w_img
    boxes[:,:,:,3] *= h_img

    for i in range(2):
    	for j in range(20):
    		probs[:,:,i,j] = numpy.multiply(class_probs[:,:,j],scales[:,:,i])
    #print (probs)
    filter_mat_probs = numpy.array(probs>=threshold,dtype='bool')
    filter_mat_boxes = numpy.nonzero(filter_mat_probs)
    boxes_filtered = boxes[filter_mat_boxes[0],filter_mat_boxes[1],filter_mat_boxes[2]]
    probs_filtered = probs[filter_mat_probs]
    classes_num_filtered = numpy.argmax(probs,axis=3)[filter_mat_boxes[0],filter_mat_boxes[1],filter_mat_boxes[2]]

    argsort = numpy.array(numpy.argsort(probs_filtered))[::-1]
    boxes_filtered = boxes_filtered[argsort]
    probs_filtered = probs_filtered[argsort]
    classes_num_filtered = classes_num_filtered[argsort]

    for i in range(len(boxes_filtered)):
    	if probs_filtered[i] == 0 : continue
    	for j in range(i+1,len(boxes_filtered)):
    		if self.iou(boxes_filtered[i],boxes_filtered[j]) > iou_threshold :
    			probs_filtered[j] = 0.0

    filter_iou = numpy.array(probs_filtered>0.0,dtype='bool')
    boxes_filtered = boxes_filtered[filter_iou]
    probs_filtered = probs_filtered[filter_iou]
    classes_num_filtered = classes_num_filtered[filter_iou]

    result = []
    for i in range(len(boxes_filtered)):
    	result.append([self.classes[classes_num_filtered[i]],boxes_filtered[i][0],boxes_filtered[i][1],boxes_filtered[i][2],boxes_filtered[i][3],probs_filtered[i]])

    return result

  def iou(self,box1,box2):
    tb = min(box1[0]+0.5*box1[2],box2[0]+0.5*box2[2])-max(box1[0]-0.5*box1[2],box2[0]-0.5*box2[2])
    lr = min(box1[1]+0.5*box1[3],box2[1]+0.5*box2[3])-max(box1[1]-0.5*box1[3],box2[1]-0.5*box2[3])
    if tb < 0 or lr < 0 : intersection = 0
    else : intersection =  tb*lr
    return intersection / (box1[2]*box1[3] + box2[2]*box2[3] - intersection)
