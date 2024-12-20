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

from helper_files.trainingeval import (iteration_logging, epoch_logging,
                          aftertraining_logging, save_predictions,
                          create_logfile)
from helper_files.trainingeval import compute_per_class_mae, compute_selfentropy_for_mae
from helper_files.resnet34 import BasicBlock
from helper_files.dataset import levels_from_labelbatch
from helper_files.losses import loss_conditional_v2
from helper_files.helper import set_all_seeds, set_deterministic
from helper_files.plotting import plot_training_loss, plot_mae, plot_accuracy
from helper_files.plotting import plot_per_class_mae
from helper_files.dataset import get_labels_from_loader
from helper_files.parser import parse_cmdline_args
from helper_files.dataset import proba_to_label


# Argparse helper
parser = argparse.ArgumentParser()
args = parse_cmdline_args(parser)

##########################
# Settings and Setup
##########################

NUM_WORKERS = args.numworkers
LEARNING_RATE = args.learningrate
NUM_EPOCHS = args.epochs
BATCH_SIZE = args.batchsize
SKIP_TRAIN_EVAL = args.skip_train_eval
SAVE_MODELS = args.save_models

if args.cuda >= 0 and torch.cuda.is_available():
    DEVICE = torch.device(f'cuda:{args.cuda}')
else:
    DEVICE = torch.device('cpu')

if args.seed == -1:
    RANDOM_SEED = None
else:
    RANDOM_SEED = args.seed

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
        'random seed': RANDOM_SEED,
        'learning rate': LEARNING_RATE,
        'num epochs': NUM_EPOCHS,
        'batch size': BATCH_SIZE,
        'output path': PATH,
        'training logfile': os.path.join(PATH, 'training.log')}
}

create_logfile(info_dict)

# Deterministic CUDA & cuDNN behavior and random seeds
#set_deterministic()
set_all_seeds(RANDOM_SEED)


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

entropy_weight = 0.5

for epoch in range(1, NUM_EPOCHS+1):

    model.train()
    for batch_idx, (features, targets) in enumerate(train_loader):

        features = features.to(DEVICE)
        targets = targets.to(DEVICE)

        # FORWARD AND BACK PROP
        logits, probas = model(features)

        # ### Ordinal loss
        loss = loss_conditional_v2(logits, targets, NUM_CLASSES)
        # ##--------------------------------------------------------------------###

        # ### Entropy term ###
        # Calculate the entropy of the batch predictions
        # # Ensure probas are clipped to avoid log(0) and ensure valid probabilities
        # probas_clipped = torch.clamp(torch.cumprod(probas, dim=1), min=1e-8, max=1.0)
        # predicted_labels = proba_to_label(probas_clipped).float()
        # batch_entropy = -torch.sum(predicted_labels * torch.log(predicted_labels)).mean()




        predicted_labels = proba_to_label(torch.cumprod(probas, dim=1)).to(DEVICE)  # Assuming proba_to_label outputs class labels
        # print('predicted_labels', predicted_labels.size(), predicted_labels)

        # Calculate the distribution of class labels in the batch
        # print('num_classes', NUM_CLASSES)
        label_counts = torch.bincount(predicted_labels, minlength=NUM_CLASSES).float()
        # print('label_counts', label_counts)
        label_distribution = label_counts / label_counts.sum()  # Normalize to get probabilities
        # Clip to avoid log(0) issues
        label_distribution = torch.clamp(label_distribution, min=1e-8)
        # print('label_distribution', label_distribution)
        
        # Calculate entropy of the predicted class labels
        batch_entropy = -torch.sum(label_distribution * torch.log(label_distribution))

        
        # Add the weighted entropy term to the loss
        
        # print(f'Before loss = {loss}')
        # print(f'Entropy = {batch_entropy}')
        # print(f'Zarib = {(1 + entropy_weight * (torch.log(torch.tensor(NUM_CLASSES)) - batch_entropy))}')
        loss = loss * (1 + entropy_weight * (torch.log(torch.tensor(NUM_CLASSES)) - batch_entropy))
        # ##--------------------------------------------------------------------###
        # print(f'After loss = {loss}\n')
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # ITERATION LOGGING
        iteration_logging(info_dict=info_dict, batch_idx=batch_idx,
                          loss=loss, train_dataset=train_dataset,
                          frequency=50, epoch=epoch)

    # EPOCH LOGGING
    # function saves best model as best_model.pt
    best_mae = epoch_logging(info_dict=info_dict,
                             model=model, train_loader=train_loader,
                             valid_loader=valid_loader,
                             which_model='conditional',
                             loss=loss, epoch=epoch, start_time=start_time,
                             skip_train_eval=SKIP_TRAIN_EVAL)

    if args.scheduler:
        scheduler.step(info_dict['training']['epoch valid rmse'][-1])


# ####### AFTER TRAINING EVALUATION
# function saves last model as last_model.pt
info_dict['last'] = {}
aftertraining_logging(model=model, which='last', info_dict=info_dict,
                      train_loader=train_loader,
                      valid_loader=valid_loader, test_loader=test_loader,
                      which_model='conditional',
                      start_time=start_time)

info_dict['best'] = {}
aftertraining_logging(model=model, which='best', info_dict=info_dict,
                      train_loader=train_loader,
                      valid_loader=valid_loader, test_loader=test_loader,
                      which_model='conditional',
                      start_time=start_time)

# ######### MAKE PLOTS ######
plot_training_loss(info_dict=info_dict, averaging_iterations=100)
plot_mae(info_dict=info_dict)
plot_accuracy(info_dict=info_dict)

# ######### PER-CLASS MAE PLOT #######

train_loader = DataLoader(dataset=train_dataset,
                          batch_size=BATCH_SIZE,
                          shuffle=False,
                          drop_last=False,
                          num_workers=NUM_WORKERS)

for best_or_last in ('best', 'last'):

    model.load_state_dict(torch.load(
        os.path.join(info_dict['settings']['output path'], f'{best_or_last}_model.pt')))

    names = {0: 'train',
             1: 'test'}
    for i, data_loader in enumerate([train_loader, test_loader]):

        true_labels = get_labels_from_loader(data_loader)

        # ######### SAVE PREDICTIONS ######
        all_probas, all_predictions = save_predictions(model=model,
                                                       which=best_or_last,
                                                       which_model='conditional',
                                                       info_dict=info_dict,
                                                       data_loader=data_loader,
                                                       prefix=names[i])

        errors, counts = compute_per_class_mae(actual=true_labels.numpy(),
                                               predicted=all_predictions.numpy())

        info_dict[f'per-class mae {names[i]} ({best_or_last} model)'] = errors

        #actual_selfentropy_best, best_selfentropy_best =\
        #    compute_selfentropy_for_mae(errors_best)

        #info_dict['test set mae self-entropy'] = actual_selfentropy_best.item()
        #info_dict['ideal test set mae self-entropy'] = best_selfentropy_best.item()

plot_per_class_mae(info_dict)

# ######## CLEAN UP ########
json.dump(info_dict, open(os.path.join(PATH, 'info_dict.json'), 'w'), indent=4)

if not SAVE_MODELS:
    os.remove(os.path.join(PATH, 'best_model.pt'))
    os.remove(os.path.join(PATH, 'last_model.pt'))
