# Imports
import os
import sys
import json
import time
import torch
import argparse
import torchvision
import numpy as np
import pandas as pd
import torch.nn as nn
import torchvision.models as models

from tqdm import tqdm
from torch.utils.data import DataLoader
from torch.utils.data import SubsetRandomSampler


# ### from local .py files

from helper_files.my_trainingeval import (iteration_logging, epoch_logging, aftertraining_logging,
                                       save_predictions, create_logfile, compute_per_class_mae,
                                       compute_selfentropy_for_mae)
from helper_files.resnet34 import BasicBlock
from helper_files.dataset import get_labels_from_loader, levels_from_labelbatch
from helper_files.losses import loss_conditional_v2
from helper_files.helper import set_all_seeds, set_deterministic
from helper_files.plotting import plot_training_loss, plot_mae, plot_accuracy, plot_per_class_mae
from helper_files.my_parser import parse_cmdline_args


# Argparse helper
parser = argparse.ArgumentParser()
args = parse_cmdline_args(parser)

##########################
# Settings and Setup
##########################

if args.cuda >= 0 and torch.cuda.is_available():
    DEVICE = torch.device(f'cuda:{args.cuda}')
else:
    DEVICE = torch.device('cpu')

if args.seed == -1:
    RANDOM_SEED = None
else:
    RANDOM_SEED = args.seed

if not os.path.exists(args.outpath):
    os.mkdir(args.outpath)

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
        'model': args.model,
        'loss': args.loss,
        'learning rate': args.learningrate,
        'num epochs': args.epochs,
        'batch size': args.batchsize,
        'output path': args.outpath,
        'training logfile': os.path.join(args.outpath, 'training.log'),
        'num workers': args.numworkers,
        'save models': args.save_models,
        'optimizer': args.optimizer,
        'scheduler': args.scheduler,
        'dataset': args.dataset,
        # 'dataset img path': args.dataset_img_path,
        # 'dataset train csv path': args.dataset_train_csv_path,
        # 'dataset valid csv path': args.dataset_valid_csv_path,
        # 'dataset test csv path': args.dataset_test_csv_path,
        'skip train eval': args.skip_train_eval}
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
                              batch_size=args.batchsize,
                              shuffle=False,  # SubsetRandomSampler shuffles
                              drop_last=True,
                              num_workers=args.numworkers,
                              sampler=train_sampler)

    valid_loader = DataLoader(dataset=valid_dataset,
                              batch_size=args.batchsize,
                              shuffle=False,
                              num_workers=args.numworkers,
                              sampler=valid_sampler)

    test_loader = DataLoader(dataset=test_dataset,
                             batch_size=args.batchsize,
                             shuffle=False,
                             num_workers=args.numworkers)

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
                              batch_size=args.batchsize,
                              shuffle=True,
                              drop_last=True,
                              num_workers=args.numworkers)

    valid_loader = DataLoader(dataset=valid_dataset,
                              batch_size=args.batchsize,
                              shuffle=False,
                              num_workers=args.numworkers)

    test_loader = DataLoader(dataset=test_dataset,
                             batch_size=args.batchsize,
                             shuffle=False,
                             num_workers=args.numworkers)

info_dict['dataset'] = DATASET_INFO
info_dict['settings']['num classes'] = NUM_CLASSES

which_model_map = {'conditional': 'conditional',
                   'crossentropy': 'categorical',
                   'mae': 'metric',
                   'conditional-entropy': 'conditional'}


##########################
# MODEL
##########################


class ResNet(nn.Module):
    def __init__(self, num_classes, grayscale):
        super(ResNet, self).__init__()
        self.num_classes = num_classes
        in_dim = 1 if grayscale else 3  # Adjust input dimension for grayscale or RGB
        if args.model == 'resnet34':
            self.resnet = models.resnet34()
        in_features = self.resnet.fc.in_features  # Get input features for fully connected layer
        if args.loss == 'conditional':
            self.resnet.fc = nn.Linear(in_features, num_classes - 1)  # Adjust output layer size
        elif args.loss == 'crossentropy':
            self.resnet.fc = nn.Linear(in_features, num_classes)  # Adjust output layer size
        elif args.loss == 'mae':
            self.resnet.fc = nn.Linear(in_features, 1)  # Adjust output layer size


    def forward(self, x):
        logits = self.resnet(x)  # Get logits from ResNet
        if args.loss == 'conditional':
            probas = torch.sigmoid(logits)  # Convert logits to probabilities using sigmoid
        elif args.loss == 'crossentropy':
            probas = nn.functional.softmax(logits)
        
        if args.loss == 'mae':
            return logits, logits
        else:
            return logits, probas


###########################################
# Initialize Cost, Model, and Optimizer
###########################################

model = ResNet(num_classes=NUM_CLASSES, grayscale=GRAYSCALE)

model.to(DEVICE)

if args.optimizer == 'adam':
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learningrate)
elif args.optimizer == 'sgd':
    optimizer = torch.optim.SGD(model.parameters(), lr=args.learningrate,
                                momentum=0.9)
else:
    raise ValueError('--optimizer must be "adam" or "sgd"')

if args.loss == 'conditional':
    criterion = loss_conditional_v2
elif args.loss == 'crossentropy':
    criterion = nn.CrossEntropyLoss()
elif args.loss == 'mae':
    criterion = nn.L1Loss()
else:
    raise ValueError('--loss invalid loss function')

if args.scheduler:
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min',
                                                           verbose=True)


start_time = time.time()

best_mae, best_rmse, best_epoch = 999, 999, -1

info_dict['training'] = {
         'num epochs': args.epochs,
         'iter per epoch': len(train_loader),
         'minibatch loss': [],
         'epoch train mae': [],
         'epoch train rmse': [],
         'epoch train acc': [],
         'epoch train loss': [],
         'epoch valid mae': [],
         'epoch valid rmse': [],
         'epoch valid acc': [],
         'epoch valid loss': [],
         'best running mae': np.inf,
         'best running rmse': np.inf,
         'best running acc': 0.,
         'best running loss': np.inf,
         'best running epoch': -1
}

for epoch in range(1, args.epochs + 1):

    model.train()
    # Initialize tqdm progress bar with only one instance
    train_bar = tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}", position=0, leave=True)
    
    for batch_idx, (features, targets) in enumerate(train_bar):

        features = features.to(DEVICE)
        targets = targets.to(DEVICE)

        # FORWARD AND BACK PROP
        logits, probas = model(features)
        
        if args.loss == 'conditional':
            loss = criterion(logits, targets, NUM_CLASSES)
        elif args.loss == 'crossentropy':
            loss = criterion(logits, targets)
        elif args.loss == 'mae':
            loss = criterion(probas, targets.view(-1, 1))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Update the tqdm progress bar in place with the current loss
        if batch_idx % 10 == 0:
            train_bar.set_postfix({"loss": f'{loss.item():.4f}'})

        # ITERATION LOGGING (if needed)
        iteration_logging(info_dict=info_dict, batch_idx=batch_idx,
                          loss=loss, train_dataset=train_dataset,
                          frequency=50, epoch=epoch)

    # EPOCH LOGGING
    # function saves best model as best_model.pt
    best_mae = epoch_logging(info_dict=info_dict,
                             model=model, train_loader=train_loader,
                             valid_loader=valid_loader,
                             which_model=which_model_map[args.loss],
                             epoch=epoch, start_time=start_time,
                             NUM_CLASSES=NUM_CLASSES,
                             skip_train_eval=args.skip_train_eval)

    if args.scheduler:
        scheduler.step(info_dict['training']['epoch valid rmse'][-1])


# ####### AFTER TRAINING EVALUATION
# function saves last model as last_model.pt
info_dict['last'] = {}
aftertraining_logging(model=model, which='last', info_dict=info_dict,
                      train_loader=train_loader,
                      valid_loader=valid_loader, test_loader=test_loader,
                      which_model=which_model_map[args.loss],
                      start_time=start_time)

info_dict['best'] = {}
aftertraining_logging(model=model, which='best', info_dict=info_dict,
                      train_loader=train_loader,
                      valid_loader=valid_loader, test_loader=test_loader,
                      which_model=which_model_map[args.loss],
                      start_time=start_time)

# ######### MAKE PLOTS ######
plot_training_loss(info_dict=info_dict, averaging_iterations=100)
plot_mae(info_dict=info_dict)
plot_accuracy(info_dict=info_dict)

# ######### PER-CLASS MAE PLOT #######

train_loader = DataLoader(dataset=train_dataset,
                          batch_size=args.batchsize,
                          shuffle=False,
                          drop_last=False,
                          num_workers=args.numworkers)

for best_or_last in ('best', 'last'):

    model.load_state_dict(torch.load(
        os.path.join(info_dict['settings']['output path'], f'{best_or_last}_model.pt'),
        weights_only=True))

    names = {0: 'train',
             1: 'test'}
    for i, data_loader in enumerate([train_loader, test_loader]):

        true_labels = get_labels_from_loader(data_loader)

        # ######### SAVE PREDICTIONS ######
        all_probas, all_predictions = save_predictions(model=model,
                                                       which=best_or_last,
                                                       which_model=which_model_map[args.loss],
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
json.dump(info_dict, open(os.path.join(args.outpath, 'info_dict.json'), 'w'), indent=4)

if not args.save_models:
    os.remove(os.path.join(args.outpath, 'best_model.pt'))
    os.remove(os.path.join(args.outpath, 'last_model.pt'))
