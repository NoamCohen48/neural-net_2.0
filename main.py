# -*- coding: utf-8 -*-
# @Author: Boyuan Jiang
# @Date  : 11/25/18

import numpy as np
import struct
from glob import glob

import csv_manager
from Layers import *
from layer.softmax_loss import Softmax_and_Loss
from utils.average import AverageMeter

def load_mnist(path, kind='train'):
    """Load MNIST data from `path`"""
    images_path = glob('./%s/%s*3-ubyte' % (path, kind))[0]
    labels_path = glob('./%s/%s*1-ubyte' % (path, kind))[0]

    with open(labels_path, 'rb') as lbpath:
        magic, n = struct.unpack('>II',
                                 lbpath.read(8))
        labels = np.fromfile(lbpath,
                             dtype=np.uint8)

    with open(images_path, 'rb') as imgpath:
        magic, num, rows, cols = struct.unpack('>IIII',
                                               imgpath.read(16))
        images = np.fromfile(imgpath,
                             dtype=np.uint8).reshape(len(labels), 784)

    return images, labels



if __name__ == '__main__':
    images, labels = csv_manager.read_file2("data/train.csv")
    # images = np.random.random((128*2, 28, 28))
    # labels = np.random.randint(0, 9, 128*2)
    labels = labels.reshape(-1).astype(int)
    test_images, test_labels = None, None
    np.random.seed(2019)

    batch_size = 200
    # Define network
    fc1 = FullyConnected(3072, 1024)
    relu4 = ReLU()
    dp = Dropout(0.9)
    fc2 = FullyConnected(1024, 10)
    sf = Softmax_and_Loss()
    val_loss = AverageMeter()
    val_acc = AverageMeter()
    init_lr = 0.1
    step=0

    for epoch in range(4):
        # train
        train_acc = 0
        train_loss = 0
        for i in range(images.shape[0] // batch_size):
            step+=1
            lr = init_lr*0.9**(step//50)
            # forward
            batch = images[i * batch_size:(i + 1) * batch_size]
            label = labels[i * batch_size:(i + 1) * batch_size]
            out = batch
            out = fc1.forward(out)
            out, relu4_cache = relu4.forward(out)
            # out, dp_cache = dp.forward(out)

            out = fc2.forward(out)
            loss, dx =  sf.forward_and_backward(out, np.array(label))

            predicted_labels = np.argmax(dx, axis=1)
            print(f"predicted: {predicted_labels}")
            accurcy = np.sum(predicted_labels == label)

            # calculate gradient
            dx = fc2.backward(dx)

            # dx = dp.gradient(dx, dp_cache)
            dx = relu4.gradient(dx, relu4_cache)
            dx = fc1.backward(dx)

            fc1.update(lr)
            fc2.update(lr)


            print(i,lr, loss, accurcy/batch_size * 100)

            if False:
                val_acc.reset()
                val_loss.reset()
                for k in range(test_images.shape[0] // batch_size):
                    batch_acc = 0
                    img = test_images[k * batch_size:(k + 1) * batch_size].reshape([batch_size, 28, 28, 1]).transpose(0, 3, 1, 2)
                    img = np.pad(img, [[0, 0], [0, 0], [2, 2], [2, 2]], mode='constant')
                    img = img / 127.5 - 1.0
                    label = test_labels[k * batch_size:(k + 1) * batch_size]

                    out, conv1_cache = conv1.forward(img)
                    out, bn1_cache = bn1.forward(out, False)
                    out, relu1_cache = relu1.forward(out)
                    out, pool1_cache = pool1.forward(out)

                    out, conv2_cache = conv2.forward(out)
                    out, bn2_cache = bn2.forward(out, False)
                    out, relu2_cache = relu2.forward(out)
                    out, pool2_cache = pool2.forward(out)

                    out, conv3_cache = conv3.forward(out)
                    out, bn3_cache = bn3.forward(out, False)
                    out, relu3_cache = relu3.forward(out)

                    conv_out = out

                    out = conv_out.reshape(batch_size, -1)
                    out, fc1_cache = fc1.forward(out)
                    out, relu4_cache = relu4.forward(out)
                    out = dp.forward(out, False)

                    out, fc2_cache = fc2.forward(out)
                    loss, dx = sf.forward_and_backward(out, np.array(label))

                    pred = np.argmax(out, axis=1)
                    correct = pred.__eq__(label).sum()
                    val_acc.update(correct/label.size*100)
                    val_loss.update(loss)
                print("val acc:", val_acc.avg, "val loss:", val_loss.avg)