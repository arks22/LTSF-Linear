import numpy as np
import torch
import matplotlib.pyplot as plt
import time
import random
import seaborn as sns
import pandas
from sklearn.metrics import confusion_matrix

plt.switch_backend('agg')

def adjust_learning_rate(optimizer, epoch, args):
    # lr = args.learning_rate * (0.2 ** (epoch // 2))
    if args.lradj == 'type1':
        lr_adjust = {epoch: args.learning_rate * (0.5 ** ((epoch - 1) // 1))}
    elif args.lradj == 'type2':
        lr_adjust = {
            2: 5e-5, 4: 1e-5, 6: 5e-6, 8: 1e-6,
            10: 5e-7, 15: 1e-7, 20: 5e-8
        }
    elif args.lradj == '3':
        lr_adjust = {epoch: args.learning_rate if epoch < 10 else args.learning_rate*0.1}
    elif args.lradj == '4':
        lr_adjust = {epoch: args.learning_rate if epoch < 15 else args.learning_rate*0.1}
    elif args.lradj == '5':
        lr_adjust = {epoch: args.learning_rate if epoch < 25 else args.learning_rate*0.1}
    elif args.lradj == '6':
        lr_adjust = {epoch: args.learning_rate if epoch < 5 else args.learning_rate*0.1}  
    if epoch in lr_adjust.keys():
        lr = lr_adjust[epoch]
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        print('Updating learning rate to {}'.format(lr))


class EarlyStopping:
    def __init__(self, patience=7, verbose=False, delta=0):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta

    def __call__(self, val_loss, model, path):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
            self.counter = 0

    def save_checkpoint(self, val_loss, model, path):
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), path + '/' + 'checkpoint.pth')
        self.val_loss_min = val_loss


class dotdict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


class StandardScaler():
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def transform(self, data):
        return (data - self.mean) / self.std

    def inverse_transform(self, data):
        return (data * self.std) + self.mean


def plot_chart(gt, pd=None, seq_len=100, pd_len=10, filename='./pic/test.pdf'):
    """
    Results visualization
    """
    plt.figure(figsize=(60,20))
    if pd is not None:
        plt.plot(pd, label='Prediction', linewidth=1)

    plt.plot(gt, label='GroundTruth', linewidth=1)

    plt.axvspan(seq_len, seq_len + pd_len, color="blue", alpha=0.1)
    plt.legend()
    plt.grid()
    plt.xlabel('date')
    plt.ylabel('price')
    plt.savefig(filename, bbox_inches='tight')


def plot_cm(gt, pd, title, index, columns, filename):
    cm = confusion_matrix(gt, pd)
    cm = pandas.DataFrame(data=cm, index=index, columns=columns)
    plt.figure(figsize=(8,8))
    sns.heatmap(cm, square=True, annot=True, fmt='d', cmap='Blues')
    plt.title(title)
    plt.xlabel("Prediction")
    plt.ylabel("GT")
    plt.savefig(filename)


def plot_scatter(gt, pd, sampling, title, r, filename):
    gt = np.ravel(gt)
    pd = np.ravel(pd)
    stacked_array = np.dstack((gt,pd))[0]

    if sampling:
        sample = np.array(random.sample(list(stacked_array), 10000)).T
    else:
        sample = stacked_array.T

    plt.figure(figsize=(15,15))
    plt.scatter(sample[0], sample[1],alpha=0.3)
    plt.grid()
    if r: 
        plt.title(title + ' | r = ' + str(r))
    plt.axhline(y=0)
    plt.axvline(x=0)
    plt.xlabel('GT price')
    plt.ylabel('predict price')
    plt.savefig(filename)


def plot_heatmap(gt, pd, sampling, title, filename):
    gt = np.ravel(gt)
    pd = np.ravel(pd)
    stacked_array = np.dstack((gt,pd))[0]

    if sampling:
        sample = np.array(random.sample(list(stacked_array), 10000)).T
    else:
        sample = stacked_array.T

    plt.figure(figsize=(15,15))
    sns.jointplot(x=sample[0], y=sample[1], kind="hist", color="#D91887")
    plt.grid()
    plt.xlabel('GT price')
    plt.ylabel('predict price')
    plt.savefig(filename)


def test_params_flop(model,x_shape):
    """
    If you want to thest former's flop, you need to give default value to inputs in model.forward(), the following code can only pass one argument to forward()
    """
    model_params = 0
    for parameter in model.parameters():
        model_params += parameter.numel()
        print('INFO: Trainable parameter count: {:.2f}M'.format(model_params / 1000000.0))
    from ptflops import get_model_complexity_info    
    with torch.cuda.device(0):
        macs, params = get_model_complexity_info(model.cuda(), x_shape, as_strings=True, print_per_layer_stat=True)
        # print('Flops:' + flops)
        # print('Params:' + params)
        print('{:<30}  {:<8}'.format('Computational complexity: ', macs))
        print('{:<30}  {:<8}'.format('Number of parameters: ', params))
