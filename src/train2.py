from __future__ import print_function
import datetime
import random
import cv2
import sys
import os
import net_sphere
from matlab_cp2tform import get_similarity_transform_for_cv2
import numpy as np
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
torch.backends.cudnn.bencmark = True
#from dataset import ImageDataset
parser = argparse.ArgumentParser(description='PyTorch sphereface')
parser.add_argument('--net', '-n', default='sphere20a', type=str)
parser.add_argument('--dataset', default='LFW', type=str)
parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
parser.add_argument('--bs', default=64, type=int, help='')
args = parser.parse_args()
use_cuda = torch.cuda.is_available()


def printoneline(*argv):
    s = ''
    for arg in argv:
        s += str(arg) + ' '
    s = s[:-1]
    sys.stdout.write('\r'+s)
    sys.stdout.flush()


def save_model(model, filename):
    state = model.state_dict()
    for key in state:
        state[key] = state[key].clone().cpu()
    torch.save(state, filename)


def dt():
    return datetime.datetime.now().strftime('%H:%M:%S')
# Returns a list of each unique image from pairsDevTrain/Test.txt


def getUniqueNameIndexTuples(imageListFile):
    # The names of the person toupled with the index of the image within the file e.g. {(Alex, 2), (Alex, 3), (Oprah, 1)...}
    uniqueNameIndexTuples = set()
    with open(imageListFile) as f:
        lines = f.readlines()
    for line in lines:  # For every line in pairsDevTest/Train.txt, convert the name and number into a filepath
        line = line[:-1]
        lineSplit = line.split('\t')
#        print(line)
 #       print(len(lineSplit))
        # This only works for the lwf dataset, and the specific pairsDevTest/Train.txt dataset
        # There are a few lines with a single number, skip them
        if (len(lineSplit) == 3):  # 1 name and 2 numbers
            uniqueNameIndexTuples.add((lineSplit[0], lineSplit[1]))
            uniqueNameIndexTuples.add((lineSplit[0], lineSplit[2]))
        elif (len(lineSplit) == 4):  # 2 names and 2 numbers
            uniqueNameIndexTuples.add((lineSplit[0], lineSplit[1]))
            uniqueNameIndexTuples.add((lineSplit[2], lineSplit[3]))
    # print(uniqueNameIndexTuples)
    return uniqueNameIndexTuples
# Gets images from a list of name index tuples and assigns each image a unique ID
# Resizes and aligns images as well


def getImagesWithLabels(imageroot, listOfImages):
    images = []
    imageroot = imageroot+os.path.sep
    imageType = ".jpg"
    # Dictionary where each person in pairsDevTrain has a unique ID
    personIDs = {w: i for (i, w) in enumerate(set(x[0] for x in listOfImages))}
    # where i is the ID and x is the imagae
    for i, x in enumerate(listOfImages):
        if(i % 100 == 0):
            print(f"processing image {i}")
        file = imageroot + makeFileName(x[0], x[1])+imageType
        baseImage = (cv2.imread(file)-127.5)/128
    #    baseImage = (cv2.imread(file))
   #     print(f"shape: {baseImage.shape}")
        resizedImage = cv2.resize(
            baseImage, (96, 112), interpolation=cv2.INTER_AREA)
        resizedImage = change_dimensions(resizedImage)
    #    print(f"NEW shape: {resizedImage.shape}")
        # -127/128 ensures values lie between -1 and 1 (b/c og values are 0-255)
        images.append((resizedImage, personIDs.get(x[0])))
    # print(personIDs)
    # print(images)
    return images
# Helper method for the data loading method, helps get the proper filepaths to load the data from


def makeFileName(name, number):  # returns
    # Number must have a total of 4 digits
    newNum = number
    for i in range(4-len(number)):
        newNum = "0"+newNum
    # Attach with underscore
    return name + os.path.sep + name + '_' + newNum


def train(epoch, args, imageList):
    print("TRAAAAAAAAAAAAAAAAAAAAAIN")
    # Augmentation occurs here TODO
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    batch_idx = 0
    while True:
        # (where batchsize = 64)
        # take 64 random indicies and make them into a batch
        # its associated 64 labels are input into the label
        img = np.zeros((args.bs, 3, 112, 96))  # or np.random.ranf
        # int array of length batchsize which corespond to the above images
        label = np.random.randint(2132, size=(args.bs, 1))
        for i in range(args.bs):
            rInt = random.randint(0, len(imageList)-1)
            # print(imageList[rInt][1])
            img[i] = imageList[rInt][0]
            label[i] = imageList[rInt][1]
        # converts to torch float (64 bit float)
        inputs = torch.from_numpy(img).float()
        targets = torch.from_numpy(label[:, 0]).long()
        # if use_cuda:
        #     inputs, targets = inputs.cuda(), targets.cuda()
        optimizer.zero_grad()
        outputs = net(inputs)  # this is where we feed it into model
        loss = criterion(outputs, targets)
        print(f"loss {loss}")
        loss.backward()
        optimizer.step()
        # .data deprecated. item() tensor of size one .item() is shortcut to turn it into python number
        train_loss += loss.item()
        outputs = outputs[0]  # 0=cos_theta 1=phi_theta
        predicted = outputs.argmax(1)
        total += targets.size(0)
        correct += (predicted == targets).long().cpu().sum()
    #    printoneline(dt(),'Te=%d Loss=%.4f | AccT=%.4f%% (%d/%d) %.4f %.2f %d'
    #        % (epoch,train_loss/(batch_idx+1), 100.0*correct/total, correct, total,
    #        criterion.lamb, criterion.it))
        # Next steps, data and label loading, fixing this deprecated stuff
        print(f"David is a genius, batch {batch_idx+1} completed")
        batch_idx += 1
        if(batch_idx == 10):
            break
    print('')
# tHanks to the homie for this sick change dimension method


def change_dimensions(image):
    new_img = np.zeros((len(image[0][0]), len(
        image), len(image[0])), dtype=float)
    # width, height, channels
    for i in range(0, len(image)):
        for j in range(0, len(image[0])):
            for k in range(0, len(image[0][0])):
                pixel = image[i][j][k]
                new_img[k][i][j] = pixel
    return new_img


# To EVALUATE. feature boolean existed b/c the 512 dimensional thing should be compared, compute the angle threshold,
# Compute angle between 2 outputs of 512 features. If they are the same class, they should be within threshhold, else they should be outside of it
if __name__ == '__main__':
    net = getattr(net_sphere, args.net)()
    # net.load_state_dict(torch.load('sphere20a_0.pth'))
    net.cuda()
    criterion = net_sphere.AngleLoss()
    print("getting unique name index tuples")
    uniqueImages = getUniqueNameIndexTuples(
        '/home/arjun/FaceRecognition/dataset/pairsDevTrain.txt')
    print("loading images")
    imageList = getImagesWithLabels(args.dataset, uniqueImages)  # Data loading
    print('start: time={}'.format(dt()))
    for epoch in range(0, 20):  # makes 20 passes thru data
        if epoch in [0, 10, 15, 18]:
            if epoch != 0:
                args.lr *= 0.1
            # Optimizer is reinitialized every time learning rate is reduced
            optimizer = optim.SGD(
                net.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
    #    imageList.shuffle()
        train(epoch, args, imageList)
        save_model(net, '{}_{}.pth'.format(args.net, epoch))
    print("finish")
