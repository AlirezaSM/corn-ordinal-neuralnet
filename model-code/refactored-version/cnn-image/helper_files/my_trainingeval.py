import os
import time
import torch
import numpy as np
import torch.nn as nn
from helper_files.dataset import proba_to_label
from helper_files.losses import loss_conditional_v2


def compute_mae_and_mse(model, data_loader, device, which_model):

    with torch.no_grad():

        mae, mse, num_examples = 0., 0., 0

        for i, (features, targets) in enumerate(data_loader):

            features = features.to(device)
            targets = targets.float().to(device)

            logits, probas = model(features)

            if which_model == 'ordinal':
                predicted_labels = proba_to_label(probas).float()
            elif which_model == 'conditional':
                probas = torch.cumprod(probas, dim=1)
                predicted_labels = proba_to_label(probas).float()
            elif which_model == 'conditional-argmax':
                ones = torch.ones((probas.shape[0], 1)).to(device)
                comp_1 = torch.cat((ones, torch.cumprod(probas, dim=1)), dim=1)
                comp_2 = torch.cat((1-probas, ones), dim=1)
                probas_y = torch.mul(comp_1, comp_2)
                predicted_labels = torch.argmax(probas_y, dim=1)
            elif which_model == 'categorical':
                _, predicted_labels = torch.max(probas, 1)
            elif which_model == 'metric':
                predicted_labels = torch.round(logits).long().view(-1)
            else:
                raise ValueError('Invalid which_model choice')

            num_examples += targets.size(0)
            mae += torch.sum(torch.abs(predicted_labels - targets))
            mse += torch.sum((predicted_labels - targets)**2)

        mae = mae / num_examples
        mse = mse / num_examples
        return mae, mse


def compute_accuracy(model, data_loader, device, which_model):

    with torch.no_grad():

        correct_pred, num_examples = 0, 0

        for i, (features, targets) in enumerate(data_loader):

            features = features.to(device)
            targets = targets.float().to(device)

            logits, probas = model(features)

            if which_model == 'ordinal':
                predicted_labels = proba_to_label(probas).float()
            elif which_model == 'conditional':
                probas = torch.cumprod(probas, dim=1)
                predicted_labels = proba_to_label(probas).float()
            elif which_model == 'conditional-argmax':
                ones = torch.ones((probas.shape[0], 1)).to(device)
                comp_1 = torch.cat((ones, torch.cumprod(probas, dim=1)), dim=1)
                comp_2 = torch.cat((1-probas, ones), dim=1)
                probas_y = torch.mul(comp_1, comp_2)
                predicted_labels = torch.argmax(probas_y, dim=1)
            elif which_model == 'categorical':
                _, predicted_labels = torch.max(probas, 1)
            elif which_model == 'metric':
                predicted_labels = torch.round(logits).long().view(-1)
            else:
                raise ValueError('invalid which_model choice')

            num_examples += targets.size(0)
            correct_pred += (predicted_labels == targets).sum()
    return correct_pred.float()/num_examples * 100


def compute_accuracy_mae_mse(model, data_loader, device, which_model, NUM_CLASSES):
    with torch.no_grad():
        correct_pred, mae, mse, total_loss, num_examples = 0, 0., 0., 0., 0
        loss_fn = None

        for i, (features, targets) in enumerate(data_loader):
            features = features.to(device)
            targets = targets.float().to(device)

            logits, probas = model(features)

            # Determine predicted labels based on the model type
            if which_model == 'ordinal':
                predicted_labels = proba_to_label(probas).float()
            elif which_model == 'conditional':
                probas = torch.cumprod(probas, dim=1)
                predicted_labels = proba_to_label(probas).float()
                loss_fn = loss_conditional_v2
                batch_loss = loss_fn(logits, targets, NUM_CLASSES)  # Using logits for loss calculation
            elif which_model == 'conditional-argmax':
                ones = torch.ones((probas.shape[0], 1)).to(device)
                comp_1 = torch.cat((ones, torch.cumprod(probas, dim=1)), dim=1)
                comp_2 = torch.cat((1 - probas, ones), dim=1)
                probas_y = torch.mul(comp_1, comp_2)
                predicted_labels = torch.argmax(probas_y, dim=1)
            elif which_model == 'categorical':
                _, predicted_labels = torch.max(probas, 1)
                loss_fn = nn.CrossEntropyLoss()
                batch_loss = loss_fn(logits, targets.long().to(device))
            elif which_model == 'metric':
                predicted_labels = torch.round(logits).long().view(-1)
                loss_fn = nn.L1Loss()
                batch_loss = loss_fn(logits, targets.long().view(-1, 1))
            else:
                raise ValueError('Invalid which_model choice')

            # Update metrics
            num_examples += targets.size(0)
            correct_pred += (predicted_labels == targets).sum()
            mae += torch.sum(torch.abs(predicted_labels - targets))
            mse += torch.sum((predicted_labels - targets) ** 2)

            # Calculate and accumulate loss
            total_loss += batch_loss.item() * targets.size(0)

        # Compute mean metrics
        mae = mae / num_examples
        mse = mse / num_examples
        avg_loss = total_loss / num_examples

    # Return accuracy, MAE, MSE, and average loss
    return correct_pred.float() / num_examples * 100, mae, mse, avg_loss


def compute_per_class_mae(actual, predicted):
    uniq = np.unique(actual)
    errors = []
    counts = []
    for label in uniq:
        mask = actual == label
        mae = np.mean(np.abs(actual[mask] - predicted[mask]))
        errors.append(mae)
        counts.append(int(np.sum(mask)))
    return errors, counts


def pytorch_selfentropy(p):
    return torch.distributions.Categorical(probs=p).entropy()


def compute_selfentropy_for_mae(errors):
    num_err = len(errors)  # same as number of classes
    errors = torch.tensor(errors)
    actual = errors / torch.sum(errors)  # normalized errors
    target = torch.tensor([1/num_err] * num_err)  # ideal uniform distribution
    actual_selfentropy = pytorch_selfentropy(actual)
    best_selfentropy = pytorch_selfentropy(target)

    return actual_selfentropy, best_selfentropy


def iteration_logging(info_dict, batch_idx,
                      loss, train_dataset,
                      frequency, epoch):

    logfile = info_dict['settings']['training logfile']
    batch_size = info_dict['settings']['batch size']
    num_epochs = info_dict['settings']['num epochs']

    info_dict['training']['minibatch loss'].append(loss.item())
    # if not batch_idx % frequency:
    #     s = (f'Epoch: {epoch:03d}/{num_epochs:03d} | '
    #          f'Batch {batch_idx:04d}/'
    #          f'{len(train_dataset)//batch_size:04d} | '
    #          f'Loss: {loss:.4f}')
    #     print(s)
    #     with open(logfile, 'a') as f:
    #         f.write(f'{s}\n')


def epoch_logging(info_dict, model, epoch,
                  start_time, which_model,
                  train_loader, valid_loader,
                  NUM_CLASSES, skip_train_eval=False):

    device = info_dict['settings']['cuda device']
    path = info_dict['settings']['output path']
    logfile = info_dict['settings']['training logfile']

    model.eval()
    with torch.no_grad():
        # Get validation metrics and loss
        valid_acc, valid_mae, valid_mse, valid_loss = compute_accuracy_mae_mse(
            model, valid_loader,
            device=device,
            which_model=which_model,
            NUM_CLASSES=NUM_CLASSES)
        valid_rmse = torch.sqrt(valid_mse)

        if not skip_train_eval:
            # Get training metrics and loss
            train_acc, train_mae, train_mse, train_loss = compute_accuracy_mae_mse(
                model, train_loader,
                device=device,
                which_model=which_model,
                NUM_CLASSES=NUM_CLASSES)
            train_rmse = torch.sqrt(train_mse)

            # Log training metrics
            info_dict['training']['epoch train mae'].append(train_mae.item())
            info_dict['training']['epoch train acc'].append(train_acc.item())
            info_dict['training']['epoch train rmse'].append(train_rmse.item())
            info_dict['training']['epoch train loss'].append(train_loss)

        # Log validation metrics
        info_dict['training']['epoch valid mae'].append(valid_mae.item())
        info_dict['training']['epoch valid acc'].append(valid_acc.item())
        info_dict['training']['epoch valid rmse'].append(valid_rmse.item())
        info_dict['training']['epoch valid loss'].append(valid_loss)

        # Update best running metrics if validation RMSE improved
        if valid_rmse < info_dict['training']['best running rmse']:
            info_dict['training']['best running mae'] = valid_mae.item()
            info_dict['training']['best running rmse'] = valid_rmse.item()
            info_dict['training']['best running acc'] = valid_acc.item()
            info_dict['training']['best running loss'] = valid_loss
            info_dict['training']['best running epoch'] = epoch
            # ######### SAVE MODEL #############
            torch.save(model.state_dict(), os.path.join(path, 'best_model.pt'))

        # Logging output
        s = (f'\nMAE/RMSE/ACC/Loss: Current Train: {train_mae:.2f}/'
             f'{train_rmse:.2f}/'
             f'{train_acc:.2f}/'
             f'{train_loss:.2f} Ep. {epoch} |'
             f' Current Valid: {valid_mae:.2f}/'
             f'{valid_rmse:.2f}/'
             f'{valid_acc:.2f}/'
             f'{valid_loss:.2f} Ep. {epoch} |'
             f' Best Valid: {info_dict["training"]["best running mae"]:.2f}'
             f'/{info_dict["training"]["best running rmse"]:.2f}'
             f'/{info_dict["training"]["best running acc"]:.2f}'
             f'/{info_dict["training"]["best running loss"]:.2f}'
             f' | Ep. {info_dict["training"]["best running epoch"]}')
        print(s)
        with open(logfile, 'a') as f:
            f.write('%s\n' % s)

        # Log elapsed time
        s = f'Time elapsed: {(time.time() - start_time)/60:.2f} min\n'
        print(s)
        with open(logfile, 'a') as f:
            f.write(f'{s}\n')


def aftertraining_logging(model, which, info_dict, train_loader,
                          valid_loader, test_loader, which_model,
                          start_time=None):

    device = info_dict['settings']['cuda device']
    path = info_dict['settings']['output path']
    logfile = info_dict['settings']['training logfile']

    if which == 'last':
        torch.save(model.state_dict(), os.path.join(path, 'last_model.pt'))
        info_dict_key = 'last'
        log_key = 'Last '

    elif which == 'best':
        model.load_state_dict(torch.load(os.path.join(path, 'best_model.pt'), weights_only=True))
        info_dict_key = 'best'
        log_key = 'Best '

    else:
        raise ValueError('`which` must be "last" or "best"')

    model.eval()
    with torch.no_grad():
        train_mae, train_mse = compute_mae_and_mse(model, train_loader,
                                                   device=device,
                                                   which_model=which_model)
        valid_mae, valid_mse = compute_mae_and_mse(model, valid_loader,
                                                   device=device,
                                                   which_model=which_model)
        test_mae, test_mse = compute_mae_and_mse(model, test_loader,
                                                 device=device,
                                                 which_model=which_model)
        train_rmse, valid_rmse = torch.sqrt(train_mse), torch.sqrt(valid_mse)
        test_rmse = torch.sqrt(test_mse)

        train_acc = compute_accuracy(model, train_loader,
                                     device=device,
                                     which_model=which_model)
        valid_acc = compute_accuracy(model, valid_loader,
                                     device=device,
                                     which_model=which_model)
        test_acc = compute_accuracy(model, test_loader,
                                    device=device,
                                    which_model=which_model)

        s = (f'MAE/RMSE: | {log_key}Train: {train_mae:.2f}/{train_rmse:.2f} '
             f'| {log_key}Valid: {valid_mae:.2f}/{valid_rmse:.2f} '
             f'| {log_key}Test: {test_mae:.2f}/{test_rmse:.2f}')
        print(s)
        with open(logfile, 'a') as f:
            f.write(f'{s}\n')

        s = (f'ACC: | {log_key}Train: {train_acc:.2f} '
             f'| {log_key}Valid: {valid_acc:.2f} '
             f'| {log_key}Test: {test_acc:.2f}')
        print(s)
        with open(logfile, 'a') as f:
            f.write(f'{s}\n')

        if start_time is not None:
            s = f'Total Running Time: {(time.time() - start_time)/60:.2f} min'
            print(s)
            with open(logfile, 'a') as f:
                f.write(f'{s}\n')

        info_dict[info_dict_key]['train mae'] = train_mae.item()
        info_dict[info_dict_key]['train rmse'] = train_rmse.item()
        info_dict[info_dict_key]['train acc'] = train_acc.item()
        info_dict[info_dict_key]['valid mae'] = valid_mae.item()
        info_dict[info_dict_key]['valid rmse'] = valid_rmse.item()
        info_dict[info_dict_key]['valid acc'] = train_acc.item()
        info_dict[info_dict_key]['test mae'] = test_mae.item()
        info_dict[info_dict_key]['test rmse'] = test_rmse.item()
        info_dict[info_dict_key]['test acc'] = test_acc.item()


def create_logfile(info_dict):
    header = []
    header.append(f'This script: {info_dict["settings"]["script"]}')
    header.append(f'PyTorch Version: {info_dict["settings"]["pytorch version"]}')
    header.append(f'CUDA device: {info_dict["settings"]["cuda device"]}')
    header.append(f'CUDA version: {info_dict["settings"]["cuda version"]}')
    header.append(f'Random seed: {info_dict["settings"]["random seed"]}')
    header.append(f'Model: {info_dict["settings"]["model"]}')
    header.append(f'Loss: {info_dict["settings"]["loss"]}')
    header.append(f'Learning rate: {info_dict["settings"]["learning rate"]}')
    header.append(f'Epochs: {info_dict["settings"]["num epochs"]}')
    header.append(f'Batch size: {info_dict["settings"]["batch size"]}')
    header.append(f'Output path: {info_dict["settings"]["output path"]}')
    header.append(f'Number of workers: {info_dict["settings"]["num workers"]}')
    header.append(f'Save models: {info_dict["settings"]["save models"]}')
    header.append(f'Optimizer: {info_dict["settings"]["optimizer"]}')
    header.append(f'Scheduler: {info_dict["settings"]["scheduler"]}')
    header.append(f'Dataset: {info_dict["settings"]["dataset"]}')
    # header.append(f'Dataset image path: {info_dict["settings"]["dataset img path"]}')
    # header.append(f'Dataset train csv path: {info_dict["settings"]["dataset train csv path"]}')
    # header.append(f'Dataset valid csv path: {info_dict["settings"]["dataset valid csv path"]}')
    # header.append(f'Dataset test csv path: {info_dict["settings"]["dataset test csv path"]}')
    header.append(f'Skip train eval: {info_dict["settings"]["skip train eval"]}')

    with open(info_dict["settings"]["training logfile"], 'w') as f:
        for entry in header:
            print(entry)
            f.write(f'{entry}\n')
            f.flush()
        print()
        f.write('\n')


def save_predictions(model, which, which_model, info_dict, data_loader, prefix='test'):

    device = info_dict['settings']['cuda device']
    path = info_dict['settings']['output path']

    if which == 'last':
        model.load_state_dict(torch.load(os.path.join(path, 'last_model.pt'), weights_only=True))

    elif which == 'best':
        model.load_state_dict(torch.load(os.path.join(path, 'best_model.pt'), weights_only=True))

    else:
        raise ValueError('`which` must be "last" or "best"')

    model.eval()
    all_pred = []
    all_probas = []
    with torch.no_grad():
        for batch_idx, (features, targets) in enumerate(data_loader):

            features = features.to(device)
            logits, probas = model(features)
            all_probas.append(probas)

            if which_model == 'ordinal':
                predicted_labels = proba_to_label(probas).float()
            elif which_model == 'conditional':
                probas = torch.cumprod(probas, dim=1)
                predicted_labels = proba_to_label(probas).float()
            elif which_model == 'conditional-argmax':
                ones = torch.ones((probas.shape[0], 1)).to(device)
                comp_1 = torch.cat((ones, torch.cumprod(probas, dim=1)), dim=1)
                comp_2 = torch.cat((1-probas, ones), dim=1)
                probas_y = torch.mul(comp_1, comp_2)
                predicted_labels = torch.argmax(probas_y, dim=1)
            elif which_model == 'categorical':
                _, predicted_labels = torch.max(probas, 1)
            elif which_model == 'metric':
                predicted_labels = torch.round(logits).long()
            else:
                raise ValueError('invalid which_model choice')

            all_pred.extend(predicted_labels)

    path_test_predictions = os.path.join(
        path, f'{prefix}_predictions_{which}_model.tensor')
    path_predicted_probas = os.path.join(
        path, f'{prefix}_allprobas_{which}_model.tensor')

    all_probas = torch.cat(all_probas).to(torch.device('cpu'))
    all_pred = torch.tensor(all_pred)
    torch.save(all_probas, path_predicted_probas)
    torch.save(all_pred, path_test_predictions)

    return all_probas, all_pred