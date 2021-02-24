from __future__ import division
import time
import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import cv2
from util import *
import argparse
import os
import os.path as osp
from darknet import Darknet
import pickle as pkl
import pandas as pd
import random


def arg_parse():
    """
    Parge argument to the detection module
    """

    parser = argparse.ArgumentParser(description='YOLO v3 Detection module')
    parser.add_argument("--images", dest="images",
                        help="Image/Directory containing images to perform detection upon", default="imgs", type=str)
    parser.add_argument("--det", dest='det', help="Image / Directory to store detections to",
                        default="det", type=str)
    parser.add_argument("--bs", dest="bs", help="Batch size", default=1)
    parser.add_argument("--confidence", dest="confidence",
                        help="Object Confidence to filter predictions", default=0.5)
    parser.add_argument("--nms_thresh", dest="nms_thresh",
                        help="NMS Threshhold", default=0.4)
    parser.add_argument("--cfg", dest='cfgfile', help="Config file",
                        default="cfg/yolov3.cfg", type=str)
    parser.add_argument("--weights", dest='weightsfile', help="weightsfile",
                        default="yolov3.weights", type=str)
    parser.add_argument("--reso", dest='reso', help="Input resolution of the network. Increase to increase accuracy. Decrease to increase speed",
                        default="416", type=str)

    return parser.parse_args()


args = arg_parse()
images = args.images
batch_size = int(args.bs)
confidence = float(args.confidence)
nms_thresh = float(args.nms_thresh)
start = 0
CUDA = torch.cuda.is_available()

num_classes = 80
classes = load_classes("data/coco.names")

print("Loading network....")
model = Darknet(args.cfgfile)
model.load_weights(args.args.weightsfile)
print("Network successfully loaded")

model.net_info["height"] = args.reso
inp_dim = int(model.net_info["height"])
assert inp_dim % 32 == 0
assert inp_dim > 32

if CUDA:
    model.cuda()

# Set the model in evalution mode
# Auto dropout, batch normalization to validate. This make accuracy is higher
model.eval()

# read_dir is a checkpoint to mesure time load image
read_dir = time.time()
# Detection phase
try:
    imlist = [osp.join(osp.realpath('.'), images, img)
              for img in osp.listdir(images)]
except NotADirectoryError:
    imlist = []
    imlist.append(ops.append(osp.realpath('.'), images))
except FileNotFoundError:
    print("No file or directory with the name {}".format(images))
    exit()

# Create det folder if not exists
if not os.path.exists(args.det):
    os.makedirs(args.det)

# Using OpenCV to load the images
# load_batch is a checkpoint to measure the time read image
load_batch = time.time()
loaded_ims = [cv2.imread(x) for img in imlist]

"""
def myfunc(a, b):
  return a + b

x = map(myfunc, ('apple', 'banana', 'cherry'), ['orange', 'lemon', 'pineapple'])

result:
x = ["appleorgane", "bananalemon", "cherrypineapple"]
"""
# Pytorch variables for images
# We will using map function like the example above
# return a list with converted image tensor to pytorch
im_batches = list(map(prep_image, loaded_ims, [
                  inp_dim for x in range(len(imlist))]))

# List containing dimensions of original images
im_dim_list = [(x.shape[1], x.shape[0]) for x in loaded_ims]
im_dim_list = torch.FloatTensor(im_dim_list).repeat(1, 2)

if CUDA:
    im_dim_list = im_dim_list.cuda()


# Create Batches
leftover = 0
if (len(img_dim_list) % batch_size):
    leftover = 1

if batch_size != 1:
    num_batches = len(imlist) // batch_size + leftover
    img_batches = [torch.cat((im_batches[i*batch_size: min((i + 1)
                                                           * batch_size, len(im_batches))])) for i in range(num_batches)]


write = 0
start_det_loop = time.time()
for i, batch in enumerate(im_batches):
    # load the image
    start = time.time()
    if CUDA:
        batch = batch.cuda()

    prediction = model(Variable(batch, volatile = True), CUDA)
    prediction = write_results(prediction, confidence, num_classes, nms_thresh)
