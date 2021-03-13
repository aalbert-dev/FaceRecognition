import argparse
import os
import sys
import cv2 as cv
import random

import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from matlab_cp2tform import get_similarity_transform_for_cv2
import net_sphere
import numpy as np
"""
@author         Arjun Albert
@modified       3/13/2021
@email          aalbert@mit.edu
@description    An imeplementation of face recognition based on the work of
                SphereFace: Deep Hypersphere Embedding for Face Recognition.           
"""

"""
Parser arguments to allow for easily adjustable model choice, learning rate,
and batch size. 
"""
parser = argparse.ArgumentParser(description='SphereFace Implementation')
parser.add_argument('--net', '-n', default='sphere4a', type=str)
parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
parser.add_argument('--bs', default=64, type=int, help='')
args = parser.parse_args()

"""
Training, testing and image folder path locations.
"""
training_dataset = '/home/arjun/FaceRecognition/dataset/pairsDevTrain.txt'
test_dataset = '/home/arjun/FaceRecognition/dataset/pairsDevTest.txt'
image_dataset = '/home/arjun/FaceRecognition/dataset/lfw'
file_sep = os.path.sep

"""
A preprocessing function to convert string labels of people's faces to
integer id's representing each unique person as a class.
"""
def get_ids(labels):
    ids = {}
    ids_to_return = []
    count = 0
    for label in labels:
        if label not in ids:
            ids[label] = count
            ids_to_return.append(count)
            count += 1
        else:
            ids_to_return.append(ids[label])
    return ids_to_return


"""
A data augmentation function to convert a list of tuples into a
sequentially ordered list such as [(a, b), (c, d)] => [a, b, c, d].
"""
def make_zipped_linear(list_of_tuples):
    raw_list = []
    for a, b in list_of_tuples:
        raw_list.append(a)
        raw_list.append(b)
    return raw_list


"""
Grabs an image from the file directory given a person's name and image id.
"""
def grab_image(label, _id):
    pfx = '_000'
    if int(_id) > 99:
        pfx = '_0'
    elif int(_id) > 9:
        pfx = '_00'
    file_name = image_dataset + file_sep + label + \
        file_sep + label + pfx + _id + '.jpg'
    img = (cv.imread(file_name) - 127.5) / 128
    scaled_img = cv.resize(img, (96, 112))
    scaled_img_proper_dims = change_dimensions(scaled_img)
    return scaled_img_proper_dims


"""
Loads a training or testing dataset and also applies various preprocessing and
data augmentation methods to ensure that image pixel channel values are 
normalized from 0-255 => -127-127, also converts the image pair format to
a sequential list. 
"""
def load_dataset(path):
    images = []
    labels = []
    with open(path) as f:
        first = True
        count = 0
        for row in f:
            if first:
                first = False
                continue
            row = row.strip('\n')
            row_values = row.split('\t')
            if len(row_values) == 3:
                label = row_values[0]
                labels.append((label, label))
                img_1 = grab_image(label, row_values[1])
                img_2 = grab_image(label, row_values[2])
                images.append((img_1, img_2))
            elif len(row_values) == 4:
                label_1 = row_values[0]
                label_2 = row_values[2]
                labels.append((label_1, label_2))
                img_1 = grab_image(label_1, row_values[1])
                img_2 = grab_image(label_2, row_values[3])
                images.append((img_1, img_2))
            count += 1
            if count % 100 == 0:
                print("Loaded " + str(count) + " images")
            # if count > 100: break
    return images, labels

"""
Changes the dimensionality of an image to be channel first, such 
that [width, height, channels] => [channels, width, height].
"""
def change_dimensions(image):
    new_img = np.zeros((len(image[0][0]), len(
        image), len(image[0])), dtype=float)
    for i in range(0, len(image)):
        for j in range(0, len(image[0])):
            for k in range(0, len(image[0][0])):
                pixel = image[i][j][k]
                new_img[k][i][j] = pixel
    return new_img


"""
Method to train the neural net on the training data while also 
using batch size to feed batch size number of image label pairs 
into the net, also keep track and reports loss, batch index, and 
epoch.
"""
def train(imageList, epoch, args):
    print("training epoch: " + str(epoch))
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    batch_idx = 0
    while True:

        img = np.zeros((args.bs, 3, 112, 96))
        label = np.random.randint(2132, size=(args.bs, 1))
        for i in range(args.bs):
            rInt = random.randint(0, len(imageList)-1)
            img[i] = imageList[rInt][0]
            label[i] = imageList[rInt][1]
        inputs = torch.from_numpy(img).float()
        targets = torch.from_numpy(label[:, 0]).long()
        optimizer.zero_grad()

        outputs = net(inputs)
        loss = criterion(outputs, targets)

        print(f"loss {loss}")
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        outputs = outputs[0]
        predicted = outputs.argmax(1)
        total += targets.size(0)
        correct += (predicted == targets).long().cpu().sum()
        print(f"batch {batch_idx+1} completed")
        batch_idx += 1
        if batch_idx > 19:
            break
    print('\n')

"""
Method to test the trained neural net on the training data while also 
using batch size to feed batch size number of image label pairs into the 
trained net, also keep track and reports loss, batch index, and accuracy.
"""
def test(imageList, args):
    print("testing")
    correct = 0
    total = 0
    batch_idx = 0
    while True:

        img = np.zeros((args.bs, 3, 112, 96))
        label = np.random.randint(2132, size=(args.bs, 1))
        for i in range(args.bs):
            rInt = random.randint(0, len(imageList)-1)
            img[i] = imageList[rInt][0]
            label[i] = imageList[rInt][1]
        inputs = torch.from_numpy(img).float()
        targets = torch.from_numpy(label[:, 0]).long()

        outputs = net(inputs)
        outputs = outputs[0]
        predicted = outputs.argmax(1)
        for i in range(0, len(predicted)):
            if predicted[i] == targets[i]:
                correct += 1
        print(correct)
        total += targets.size(0)
        correct += (predicted == targets).long().cpu().sum()
        print(f"batch {batch_idx+1} completed")
        batch_idx += 1
        if batch_idx > 9:
            break
    print('\n')

"""
Import the training dataset, apply preprocessing methods and
data augmentation to get training data ready for training.
"""
train_images, train_labels = load_dataset(training_dataset)
processed_train_images = make_zipped_linear(train_images)
processed_train_labels = make_zipped_linear(train_labels)
label_ids = get_ids(processed_train_labels)
image_with_labels = []
for image, label in zip(processed_train_images, label_ids):
    image_with_labels.append((image, label))

"""
Inititalize the neural network and loss function based on 
parser argument default parameters. 
"""
net = getattr(net_sphere, args.net)()
criterion = net_sphere.AngleLoss()

"""
Run training for a set number (20) epochs with a stochastic gradient descent 
optimizer and parameters based on the argument parser. 
"""

for epoch in range(0, 10):
    if epoch in [0, 4, 8]: 
        optimizer = optim.SGD(net.parameters(), lr=args.lr,
                              momentum=0.9, weight_decay=5e-4)
    train(image_with_labels, epoch, args)

"""
Import the testing dataset, apply preprocessing methods and
data augmentation to get testing data ready for testing.
"""
test_images, test_labels = load_dataset(test_dataset)
processed_test_images = make_zipped_linear(test_images)
processed_test_labels = make_zipped_linear(test_labels)
test_label_ids = get_ids(processed_test_labels)
test_images_with_labels = []
for image, label in zip(processed_test_images, test_label_ids):
    test_images_with_labels.append((image, label))

"""
Run tests on trained model with default parameters 
"""
test(test_images_with_labels, args)

print('finished')
