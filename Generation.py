# Digital Universal Adversarial Perturbation Generation
# Victim Model: ResNet18/50/101
# Dataset: CIFAR10
# Adversarial Attack Algorithm: FGSM / DeepFool
# Created by Junbo Zhao 2020.3.4

""" Implementation of Universal adversarial perturbations in PyTorch.
Reference:
[1] Seyed-Mohsen Moosavi-Dezfooli, Alhussein Fawzi, Omar Fawzi, Pascal Frossard
    Universal adversarial perturbations. arXiv:1610.08401
[2] Jonas Rauber, Wieland Brendel, Matthias Bethge
    Foolbox: A Python toolbox to benchmark the robustness of machine learning models. arXiv:1707.04131
"""

# TODO:Nothing. Failed.\
#  While following paper[1] to attack with DeepFool, I don't know why it cannot converge on the training set.\
#  And while attacking with FGSM, a well-performed universal adversarial perturbation is found on the training set.\
#  But I don't know why the found perturbation cannot generalize to the test set.\
#  I even doubt that the algorithm cannot work on CIFAR10 at all.

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

import foolbox

def project_perturbation(radius, p, perturbation):
    if p == 2:
        perturbation = perturbation * min(1, radius / np.linalg.norm(perturbation.flatten(1)))
    elif p == np.inf:
        perturbation = np.sign(perturbation) * np.minimum(abs(perturbation), radius)
    return perturbation

def test(model, testloader, criterion, perturbation):
    correct, total, loss = 0, 0, 0
    with torch.no_grad():
        for (images, labels) in testloader:
            images = torch.from_numpy(np.clip((images.numpy() + perturbation), a_min=-1, a_max=1))
            images = images.cuda()
            labels = labels.cuda()
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            loss += criterion(outputs, labels).item()
    return loss / total, correct / total

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

parser = argparse.ArgumentParser()
parser.add_argument('--threshold', type=float, default=0.001, help="the difference threshold between the original picture and the adversarial example")
parser.add_argument('--epochs', type=int, default=20, help="total epochs")
parser.add_argument('--radius', type=float, default=0.1, help="projection radius")
parser.add_argument('--foolrate', type=float, default=0.8, help="fool rate")
parser.add_argument('--model_address', type=str, default="ResNet18.pkl", help="address of the pretrained model")
parser.add_argument('--dataset_address', type=str, default="/home/eva_share/datasets/cifar10", help="address of the dataset")
parser.add_argument('--batch_size', type=int, default=64, help="batch size")
parser.add_argument('--num_workers', type=int, default=32, help="num_workers")
parser.add_argument('--log_address', type=str, default="Generation_log.csv", help="address of the generation log")
parser.add_argument('--p', type=int, default=np.inf, help="norm")
parser.add_argument('--sample_num', type=int, default=100, help="number of the sampled images to generate the universal perturbation")
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
universal_perturbation = torch.rand(3, 32, 32)

# Setup the attack method
criterion = nn.CrossEntropyLoss()
fmodel = foolbox.models.PyTorchModel(model, bounds=(-1, 1), num_classes=10)
attack = foolbox.attacks.FGSM(fmodel, distance=foolbox.distances.MeanSquaredDistance)

# Establish the generation log
with open(str(args.p) + '_' + args.log_address, 'w') as f:
    writer = csv.writer(f)
    writer.writerow(["epoch", "train_success_rate%", "test_success_rate%", "test_loss"])

for epoch in range(args.epochs):

    # Generate the universal perturbation on the trainset
    start_time = time.time()
    train_successful_attack, train_total, original_successful_attack = 0, 0, 0
    for (images, labels) in trainloader:
        if train_total > args.sample_num:
            break
        images.cuda()
        outputs = model(images)
        _, predicts = torch.max(outputs.data, 1)
        images = images.cpu().numpy()
        labels = labels.numpy()
        train_total += images.shape[0]
        for i in range(images.shape[0]):
            # Add the universal perturbation to the original picture
            images[i] = np.clip((images[i] + universal_perturbation), a_min=-1, a_max=1)
            image = images[i][np.newaxis, :, :, :]
            new_image = image.copy()

            # Get the model's predict of the perturbed picture
            image = torch.from_numpy(image).cuda()
            output = model(image)
            _, predict = torch.max(output.data, 1)

            # If the original predict is the same as the perturbed predict, no perturbation will be added to the universal perturbation
            if predicts[i] != predict[0]:
                train_successful_attack += 1
                original_successful_attack += 1

            # If not, generate the perturbation and add it to the universal perturbation
            else:
                label = predicts[i].cpu().numpy()
                label = np.array([label])
                adversarial = attack(new_image, label, unpack=False)
                # If the attack fails, pass
                if adversarial[0].perturbed is None or adversarial[0].distance.value > args.threshold:
                    pass
                else:
                    train_successful_attack += 1
                    perturbation = adversarial[0].perturbed - adversarial[0].unperturbed
                    universal_perturbation += perturbation
                    universal_perturbation = project_perturbation(args.radius, args.p, universal_perturbation)

    # Print the statistics
    end_time = time.time()
    print("epoch:{}, Consumed Time:{}s".format(epoch, end_time - start_time))
    print("         trainset successful attack:{:.3f}% original successful attack:{:.3f}%".format(100 * train_successful_attack / train_total, 100 * original_successful_attack / train_total))

    # Test the generated perturbation on the testset
    test_loss, test_acc = test(model, testloader, criterion, universal_perturbation)
    print("         testset successful attack:{:.3f}% loss:{:.3f}".format(100 * (1 - test_acc), test_loss))

    # Save the statistics in the log
    with open(str(args.p) + "_" + args.log_address, 'a') as f:
        writer = csv.writer(f)
        writer.writerow([epoch, 100 * train_successful_attack / train_total, 100 * (1 - test_acc), test_loss])

    # Save the generated universal perturbation for every epoch
    universal_per = universal_perturbation.copy()
    universal_per = abs(universal_per)
    universal_per = (universal_per - np.min(universal_per)) / (np.max(universal_per) - np.min(universal_per))
    plt.imshow(universal_per.transpose(1, 2, 0))
    plt.savefig(str(epoch) + '_' + str(args.p) + '_universal_perturbation.png')

    # If the perturbation achieve the set fool rate, break
    if test_acc < 1 - args.foolrate:
        print("Universal Perturbation Generated!")
        break