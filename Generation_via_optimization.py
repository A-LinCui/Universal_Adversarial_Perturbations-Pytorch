# Digital Universal Perturbation Generation via Optimization
# Victim Model: ResNet18/50/101
# Dataset: CIFAR10
# Optimization Algorithm: Adam
# Created by Junbo Zhao 2020.3.4

# TODO:Nothing.This idea totally failed.\
#  It's amazing that the found universal perturbation even performs worse than the random perturbation while attacking untargetedly.\
#  And while attacking targetedly, the successful attacking rate is about 10.00%, which is the percentage of the target label pictures in the dataset,\
#  which also means failure.

import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.autograd import Variable
from torch.utils.data import DataLoader
import torch.utils.data.dataloader as Data

import os
import csv
import matplotlib.pyplot as plt
import numpy as np
import argparse
import time

from resnet import *

def test(model, testloader, criterion, perturbation, target):
    correct, total, loss = 0, 0, 0
    with torch.no_grad():
        for (images, labels) in testloader:
            if target != -1:
                labels = torch.ones_like(labels) * target
            images = images.cuda()
            labels = labels.cuda()
            images = torch.clamp((perturbation + images), 0, 1)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            loss += criterion(outputs, labels).item()
    return loss / total, correct / total

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

parser = argparse.ArgumentParser()
parser.add_argument('--threshold', type=float, default=0.04, help="the difference threshold between the original picture and the adversarial example")
parser.add_argument('--epochs', type=int, default=100, help="total epochs")
parser.add_argument('--foolrate', type=float, default=0.8, help="fool rate")
parser.add_argument('--model_address', type=str, default="ResNet18.pkl", help="address of the pretrained model")
parser.add_argument('--dataset_address', type=str, default="/home/eva_share/datasets/cifar10", help="address of the dataset")
parser.add_argument('--batch_size', type=int, default=64, help="batch size")
parser.add_argument('--num_workers', type=int, default=32, help="num_workers")
parser.add_argument('--log_address', type=str, default="Generation_log.csv", help="address of the generation log")
parser.add_argument('--target', type=int, default=-1, help="target")
args = parser.parse_args()

# Set the transformation
transform_train = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])

# Load the dataset
trainset = torchvision.datasets.CIFAR10(root=args.dataset_address, train=True, download=True, transform=transform_train)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
testset = torchvision.datasets.CIFAR10(root=args.dataset_address, train=False, download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

# Establish the victim model from pre-trained model
model = ResNet18()
model = nn.DataParallel(model)
model.load_state_dict(torch.load(args.model_address))
model.cuda()
model = nn.DataParallel(model)
model.eval()

for param in model.parameters():
    param.requires_grad = False

# Initialize the universal perturbation
universal_perturbation = Variable(torch.from_numpy(np.random.randn(3, 32, 32)).float()).cuda()
universal_perturbation.requires_grad = True

# Setup the attack method
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam([universal_perturbation], lr=1e-8)

# Test the accuracy on the testset with random perturbation
rand_acc = [0, 0, 0, 0, 0]
for i in range(5):
    perturbation = torch.randn(3, 32, 32)
    perturbation *= args.threshold / abs(perturbation).max()
    perturbation = perturbation.cuda()
    _, rand_acc[i] = test(model, testloader, criterion, perturbation, args.target)
print("Accuracy for random perturbation:{:.3f}%".format(100 * np.mean(rand_acc)))

# Establish the generation log
with open(args.log_address, 'w') as f:
    writer = csv.writer(f)
    writer.writerow(["epoch", "train_foolrate", "test_foolrate", "train_loss", "test_loss"])

for epoch in range(args.epochs):

    # Generate the universal perturbation on the trainset
    start_time = time.time()
    train_successful_attack, train_total, train_loss = 0, 0, 0

    for data in trainloader:
        images, labels = data
        if args.target != -1:
            images = torch.ones_like(images) * args.target
        images = images.cuda()
        labels = labels.cuda()
        per_images = images + universal_perturbation
        per_images = torch.clamp(per_images, 0, 1)
        outputs = model(per_images)
        _, predicts = torch.max(outputs.data, 1)
        train_total += labels.size(0)
        if args.target != -1:
            train_successful_attack += (predicts == labels).sum().item()
            loss = criterion(outputs, labels)
        else:
            train_successful_attack += (predicts != labels).sum().item()
            loss = - criterion(outputs, labels)
        optimizer.zero_grad()
        loss.backward(retain_graph=True)
        optimizer.step()
        if args.target == -1:
            train_loss -= loss.item()
        else:
            train_loss += loss.item()
        universal_perturbation = torch.clamp(universal_perturbation, - args.threshold, args.threshold)
    print("         trainset successful attack:{:.3f}% loss:{}".format(100 * train_successful_attack / train_total, train_loss / train_total))

    # Test the generated perturbation on the testset
    test_loss, test_acc = test(model, testloader, criterion, universal_perturbation, args.target)
    if args.target == -1:
        print("         testset successful attack:{:.3f}% loss:{:.3f}".format(100 * (1 - test_acc), test_loss))
    else:
        print("         testset successful attack:{:.3f}% loss:{:.3f}".format(100 * test_acc, test_loss))

    # Calculate the consumed time
    end_time = time.time()
    print("epoch:{}, Consumed Time:{}s".format(epoch, end_time - start_time))

    # Save the statistics in the log
    with open(args.log_address, 'a') as f:
        writer = csv.writer(f)
        writer.writerow([epoch, train_successful_attack / train_total, (1 - test_acc), train_loss, test_loss])

    # Save the generated universal perturbation for every epoch
    universal_per = universal_perturbation.clone().detach().cpu().numpy()
    avg_norm = np.linalg.norm(abs(universal_per)) / (universal_per.shape[0] * universal_perturbation.shape[1] * universal_perturbation.shape[2])
    print("average norm for present universal perturbation:{:.5f}".format(avg_norm))

    # Normalization
    universal_per = (universal_per - np.min(universal_per)) / (np.max(universal_per) - np.min(universal_per))
    plt.imshow(universal_per.transpose(1, 2, 0))
    plt.savefig(str(epoch) + '_' + '_universal_perturbation.png')

    # If the L2 distance has been larger than the threshold, break
    if avg_norm > args.threshold:
        print("Universal Perturbation Not Found!")
        break

    # If the perturbation achieve the set fool rate, break
    if test_acc < 1 - args.foolrate:
        print("Universal Perturbation Generated!")
        break