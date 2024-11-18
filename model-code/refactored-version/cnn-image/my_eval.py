# coding: utf-8

# Like v2, and in contrast to v1, this version removes the cumprod from the forward pass

# In addition, it uses a different conditional loss function compared to v2.
# Here, the loss is computed as the average loss of the total samples, 
# instead of firstly averaging the cross entropy inside each task and then averaging over tasks equally. 
# The weight of each task will be adjusted
# for the sample size used for training each task naturally without manually setting the weights.

# Imports

import os
import json
import pandas as pd
import time
import torch
import torch.nn as nn
import argparse
import sys
import numpy as np
import torchvision.models as models

from torch.utils.data import DataLoader
from torch.utils.data import SubsetRandomSampler


# ### from local .py files

from helper_files.my_eval import (iteration_logging, epoch_logging,
                          aftertraining_logging, save_predictions,
                          create_logfile)
from helper_files.my_eval import compute_per_class_mae, compute_selfentropy_for_mae
from helper_files.resnet34 import BasicBlock
from helper_files.dataset import levels_from_labelbatch
from helper_files.losses import loss_conditional_v2
from helper_files.helper import set_all_seeds, set_deterministic
from helper_files.plotting import plot_training_loss, plot_mae, plot_accuracy
from helper_files.plotting import plot_per_class_mae
from helper_files.dataset import get_labels_from_loader
from helper_files.parser import parse_cmdline_args


# Argparse helper
parser = argparse.ArgumentParser()
args = parse_cmdline_args(parser)

##########################
# Settings and Setup
##########################

NUM_WORKERS = args.numworkers
BATCH_SIZE = args.batchsize

if args.cuda >= 0 and torch.cuda.is_available():
    DEVICE = torch.device(f'cuda:{args.cuda}')
else:
    DEVICE = torch.device('cpu')

PATH = args.outpath

if not os.path.exists(PATH):
    os.mkdir(PATH)

cuda_device = DEVICE
if torch.cuda.is_available():
    cuda_version = torch.version.cuda
else:
    cuda_version = 'NA'

info_dict = {
    'settings': {
        'script': os.path.basename(__file__),
        'pytorch version': torch.__version__,
        'cuda device': str(cuda_device),
        'cuda version': cuda_version,
        'output path': PATH,
        'batch size': BATCH_SIZE,
        'evaluation logfile': os.path.join(PATH, 'eval.log')}
}

create_logfile(info_dict)

###################
# Dataset
###################

if args.dataset == 'mnist':
    from helper_files.constants import MNIST_INFO as DATASET_INFO
    from torchvision.datasets import MNIST as PyTorchDataset
    from helper_files.dataset import mnist_train_transform as train_transform
    from helper_files.dataset import mnist_validation_transform as validation_transform

elif args.dataset == 'morph2':
    from helper_files.constants import MORPH2_INFO as DATASET_INFO
    from helper_files.dataset import Morph2Dataset as PyTorchDataset
    from helper_files.dataset import morph2_train_transform as train_transform
    from helper_files.dataset import morph2_validation_transform as validation_transform

elif args.dataset == 'morph2-balanced':
    from helper_files.constants import MORPH2_BALANCED_INFO as DATASET_INFO
    from helper_files.dataset import Morph2Dataset as PyTorchDataset
    from helper_files.dataset import morph2_train_transform as train_transform
    from helper_files.dataset import morph2_validation_transform as validation_transform

elif args.dataset == 'afad-balanced':
    from helper_files.constants import AFAD_BALANCED_INFO as DATASET_INFO
    from helper_files.dataset import AFADDataset as PyTorchDataset
    from helper_files.dataset import afad_train_transform as train_transform
    from helper_files.dataset import afad_validation_transform as validation_transform

elif args.dataset == 'aes':
    from helper_files.constants import AES_INFO as DATASET_INFO
    from helper_files.dataset import AesDataset as PyTorchDataset
    from helper_files.dataset import aes_train_transform as train_transform
    from helper_files.dataset import aes_validation_transform as validation_transform

else:
    raise ValueError('Dataset choice not supported')

###################
# Dataset
###################

if args.dataset == 'mnist':

    NUM_CLASSES = 10
    GRAYSCALE = True
    RESNET34_AVGPOOLSIZE = 1

    train_dataset = PyTorchDataset(root='./datasets',
                                   train=True,
                                   download=True,
                                   transform=train_transform())

    valid_dataset = PyTorchDataset(root='./datasets',
                                   train=True,
                                   transform=validation_transform(),
                                   download=False)

    test_dataset = PyTorchDataset(root='./datasets',
                                  train=False,
                                  transform=validation_transform(),
                                  download=False)

    train_indices = torch.arange(1000, 60000)
    valid_indices = torch.arange(0, 1000)
    train_sampler = SubsetRandomSampler(train_indices)
    valid_sampler = SubsetRandomSampler(valid_indices)

    train_loader = DataLoader(dataset=train_dataset,
                              batch_size=BATCH_SIZE,
                              shuffle=False,  # SubsetRandomSampler shuffles
                              drop_last=True,
                              num_workers=NUM_WORKERS,
                              sampler=train_sampler)

    valid_loader = DataLoader(dataset=valid_dataset,
                              batch_size=BATCH_SIZE,
                              shuffle=False,
                              num_workers=NUM_WORKERS,
                              sampler=valid_sampler)

    test_loader = DataLoader(dataset=test_dataset,
                             batch_size=BATCH_SIZE,
                             shuffle=False,
                             num_workers=NUM_WORKERS)

else:

    GRAYSCALE = False
    RESNET34_AVGPOOLSIZE = 4
    
    if args.dataset_train_csv_path:
        DATASET_INFO['TRAIN_CSV_PATH'] = args.dataset_train_csv_path

    if args.dataset_valid_csv_path:
        DATASET_INFO['VALID_CSV_PATH'] = args.dataset_valid_csv_path
        
    if args.dataset_test_csv_path:
        DATASET_INFO['TEST_CSV_PATH'] = args.dataset_test_csv_path
        
    if args.dataset_img_path:
        DATASET_INFO['IMAGE_PATH'] = args.dataset_img_path

    df = pd.read_csv(DATASET_INFO['TRAIN_CSV_PATH'], index_col=0)
    classes = df[DATASET_INFO['CLASS_COLUMN']].values
    del df
    train_labels = torch.tensor(classes, dtype=torch.float)
    NUM_CLASSES = torch.unique(train_labels).size()[0]
    del classes

    train_dataset = PyTorchDataset(csv_path=DATASET_INFO['TRAIN_CSV_PATH'],
                                   img_dir=DATASET_INFO['IMAGE_PATH'],
                                   transform=train_transform())

    test_dataset = PyTorchDataset(csv_path=DATASET_INFO['TEST_CSV_PATH'],
                                  img_dir=DATASET_INFO['IMAGE_PATH'],
                                  transform=validation_transform())

    valid_dataset = PyTorchDataset(csv_path=DATASET_INFO['VALID_CSV_PATH'],
                                   img_dir=DATASET_INFO['IMAGE_PATH'],
                                   transform=validation_transform())

    train_loader = DataLoader(dataset=train_dataset,
                              batch_size=BATCH_SIZE,
                              shuffle=True,
                              drop_last=True,
                              num_workers=NUM_WORKERS)

    valid_loader = DataLoader(dataset=valid_dataset,
                              batch_size=BATCH_SIZE,
                              shuffle=False,
                              num_workers=NUM_WORKERS)

    test_loader = DataLoader(dataset=test_dataset,
                             batch_size=BATCH_SIZE,
                             shuffle=False,
                             num_workers=NUM_WORKERS)

info_dict['dataset'] = DATASET_INFO
info_dict['settings']['num classes'] = NUM_CLASSES


##########################
# MODEL
##########################


class ResNet(nn.Module):
    def __init__(self, num_classes, grayscale):
        super(ResNet, self).__init__()
        self.num_classes = num_classes
        in_dim = 1 if grayscale else 3  # Adjust input dimension for grayscale or RGB
        self.resnet = models.resnet34()
        in_features = self.resnet.fc.in_features  # Get input features for fully connected layer
        self.resnet.fc = nn.Linear(in_features, num_classes - 1)  # Adjust output layer size

    def forward(self, x):
        logits = self.resnet(x)  # Get logits from ResNet
        probas = torch.sigmoid(logits)  # Convert logits to probabilities using sigmoid
        return logits, probas  # Return both logits and probabilities



###########################################
# Initialize Cost, Model, and Optimizer
###########################################

model = ResNet(num_classes=NUM_CLASSES, grayscale=GRAYSCALE)

model.to(DEVICE)

if args.optimizer == 'adam':
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
elif args.optimizer == 'sgd':
    optimizer = torch.optim.SGD(model.parameters(), lr=LEARNING_RATE,
                                momentum=0.9)
else:
    raise ValueError('--optimizer must be "adam" or "sgd"')

if args.scheduler:
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min',
                                                           verbose=True)


start_time = time.time()

best_mae, best_rmse, best_epoch = 999, 999, -1

info_dict['training'] = {
         'num epochs': NUM_EPOCHS,
         'iter per epoch': len(train_loader),
         'minibatch loss': [],
         'epoch train mae': [],
         'epoch train rmse': [],
         'epoch train acc': [],
         'epoch valid mae': [],
         'epoch valid rmse': [],
         'epoch valid acc': [],
         'best running mae': np.infty,
         'best running rmse': np.infty,
         'best running acc': 0.,
         'best running epoch': -1
}

info_dict['best'] = {}
aftertraining_logging(model=model, which='best', info_dict=info_dict,
                      train_loader=train_loader,
                      valid_loader=valid_loader, test_loader=test_loader,
                      which_model='conditional',
                      start_time=start_time)