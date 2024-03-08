from options import opt
from config import set_config
from modules.model import CNNMnist, ResNet9, ResNet18, ResNet34, ResNet
from roles.FmpuTrainer import FmpuTrainer
import torch.nn as nn

import os

import torch
import random
import numpy as np


def set_seed(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False



def main():
    set_seed(opt.seed)
    set_config(opt)
    print("Acc from:", opt, "\n")

    if opt.dataset == 'MNIST':
        model = CNNMnist().cuda()
        trainer = FmpuTrainer(model)
    if opt.dataset == 'CIFAR10':
        model = ResNet9().cuda()
        trainer = FmpuTrainer(model)
    if opt.dataset == 'FMNIST':
        model = CNNMnist().cuda()
        trainer = FmpuTrainer(model)
    if opt.dataset == 'SVHN':
        model = ResNet9().cuda()
        trainer = FmpuTrainer(model)

    trainer.begin_train()


if __name__ == '__main__':
    # merge config
    print(opt.positiveRate)
    print(opt.seed)
    main()
