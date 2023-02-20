import numpy as np
import os
import pandas as pd
from scipy import special
import matplotlib.pyplot as plt
import csv
import h5py
import cvxpy as cp
import torchvision.datasets as datasets
import torch
from torchvision import transforms
from kymatio.sklearn import Scattering2D
from sklearn import preprocessing
from scipy.stats import ortho_group
from scipy.linalg import hadamard
from math import ceil
from scipy.fftpack import dct

def preprocess_data_mnist(dataset):
    X = torch.clone(dataset.data).float()
    y = torch.clone(dataset.targets).float()
    idx = np.argsort(y)
    print(y)
    X = X[idx]
    data = X.numpy()
    data = np.array(data)
    data -= data.mean(axis=0)
    data /= data.std()
    return np.array(data)

def preprocess_data_fashionmnist(dataset):
    X = torch.clone(dataset.data).float()
    y = torch.clone(dataset.targets).float()
    idx = np.argsort(y)
    print(y)
    X = X[idx]
    data = X.numpy()
    data = np.array(data)
    data -= data.mean(axis=0)
    data /= data.std()
    return np.array(data)
def get_F(dim, typeF='gaussian'):
    p, n = dim
    rank = np.min(dim)
    if typeF == 'gaussian':
        return np.random.normal(loc=0.0, scale=1.0, size=dim)
    elif typeF == 'orthogonal':
        U, V = ortho_group.rvs(p), ortho_group.rvs(n) 
        D = np.zeros(dim)
        np.fill_diagonal(D, 1)
        if p > n:
            return U @ D @ V * np.sqrt(p)
        else: 
            return U @ D @ V * np.sqrt(n)
    elif typeF == 'hadamard':
        if p >= n:
            H = hadamard(p)
            return subsample(H, p-n, kind='cols')
        elif p<n:
            H = hadamard(n)
            return subsample(H, n-p, kind='rows')
    elif typeF == 'dct':
        if n >= p:
            D = 1/np.sqrt(2) * dct(np.eye(n))
            return subsample(D, n-p, kind='rows', keep0=False)
        elif n<p:
            D = 1/np.sqrt(2) * dct(np.eye(p))
            return subsample(D, p-n, kind='cols', keep0=False)
def subsample(mat, p, kind='rows', keep0=True):
    '''
    Deletes p random rows/columns of mat, excluding the first row/column
    '''
    n_rows, n_cols = mat.shape
    if (kind=='rows') and (p<=n_rows):
        if keep0:
            probs = [0]+list(1/(n_rows-1) * np.ones(n_rows-1))
        else: 
            probs = list(1/(n_rows) * np.ones(n_rows))
        choices = np.random.choice(n_rows, p, p=probs, replace=False)
        new_mat = np.array([row for (k, row) in enumerate(mat) if k not in choices])
        
    elif (kind=='cols') and (p<=n_cols):
        if keep0:
            probs = [0]+list(1/(n_rows-1) * np.ones(n_rows-1))
        else: 
            probs = list(1/(n_rows) * np.ones(n_rows))

        choices = np.random.choice(n_cols, p, p=probs, replace=False)
        new_mat = np.array([col.T for (k, col) in enumerate(mat.T) if not k in choices]).T
    else:
        raise NameError('Can only subsample a smaller number of columns or rows!!!')
        
    return new_mat        
def preprocess_data_cifar10(dataset):
    classDict = {'plane': 0, 'car': 1, 'bird': 2, 'cat': 3, 'deer': 4,
             'dog': 5, 'frog': 6, 'horse': 7, 'ship': 8, 'truck': 9}
    n_samples, _, _, _ = dataset.data.shape
    trainloader = torch.utils.data.DataLoader(dataset, batch_size=n_samples, shuffle=True, num_workers=0)
    data = []
    for i, batch in enumerate(trainloader):
        for k,l in enumerate(batch[1]):
            data.append(batch[0][k].numpy())
    data = np.array(data)
    y = dataset.targets
    idx = np.argsort(y)
    print(y)
    data = data[idx]
    data -= data.mean(axis=0)
    data /= data.std()
    return np.array(data)[:,0,:,:]
def relu(x):
    return x * (x > 0.) + 0 * (x <= 0.)
def get_real_data(p, which_real_dataset = "MNIST", which_transform = "no_transform", which_non_lin = "none", path_to_data_folder = './'):
    if os.path.isdir(path_to_data_folder) == False: # create the data folder if not there
        try:
            os.makedirs(path_to_data_folder)
        except OSError:
            print ("\n!!! ERROR: Creation of the directory %s failed" % path_to_data_folder)
            raise
    if os.path.isfile(path_to_data_folder +  '/' + 'X_%s_%s_%s.hdf5'%(which_real_dataset, which_transform, which_non_lin)) == False:
        if which_real_dataset == "MNIST":
            if which_transform == 'no_transform':
                mnist = datasets.MNIST(root='data', train=True, download=True, transform=None)
                X = preprocess_data_mnist(mnist)
                ntot, dx, dy = X.shape
                X = X.reshape(ntot, -1)
            elif which_transform == 'other':
                mnist = datasets.MNIST(root='data', train=True, download=True, transform=None)
                X = preprocess_data_mnist(mnist)
                ntot, dx, dy = X.shape
                X = X.reshape(ntot, -1) ; ntot, d = X.shape
                F = get_F((p,d), typeF='hadamard')
                scaler = preprocessing.StandardScaler().fit(X)
                X = scaler.transform(X)
            elif which_transform == 'random_gaussian_features':
                if which_non_lin == 'erf':
                    mnist = datasets.MNIST(root='data', train=True, download=True, transform=None)
                    X = preprocess_data_mnist(mnist)
                    ntot, dx, dy = X.shape
                    X = X.reshape(ntot, -1)
                    ntot, d = X.shape
                    F = np.random.normal(0., 1., size = (d, p))
                    X = special.erf(X @ F)
                elif which_non_lin == 'tanh':
                    mnist = datasets.MNIST(root='data', train=True, download=True, transform=None)
                    X = preprocess_data_mnist(mnist)
                    ntot, dx, dy = X.shape
                    X = X.reshape(ntot, -1)
                    ntot, d = X.shape
                    F = np.random.normal(0., 1., size = (d, p))
                    X = np.tanh(X @ F)
                elif which_non_lin == 'sign':
                    mnist = datasets.MNIST(root='data', train=True, download=True, transform=None)
                    X = preprocess_data_mnist(mnist)
                    ntot, dx, dy = X.shape
                    X = X.reshape(ntot, -1)
                    ntot, d = X.shape
                    F = np.random.normal(0., 1., size = (d, p))
                    X = np.sign(X @ F)
                elif which_non_lin == 'relu':
                    mnist = datasets.MNIST(root='data', train=True, download=True, transform=None)
                    X = preprocess_data_mnist(mnist)
                    ntot, dx, dy = X.shape
                    X = X.reshape(ntot, -1)
                    ntot, d = X.shape
                    F = np.random.normal(0., 1., size = (d, p))
                    X = relu(X @ F)
        elif which_real_dataset == "fashion-MNIST":
            if which_transform == 'no_transform':
                fashion_mnist = datasets.FashionMNIST(root='data', train=True, download=True, transform=None)
                X = preprocess_data_fashionmnist(fashion_mnist)
                ntot, dx, dy = X.shape
                X = X.reshape(ntot, -1)
            elif which_transform == 'other':
                fashion_mnist = datasets.FashionMNIST(root='data', train=True, download=True, transform=None)
                X = preprocess_data_fashionmnist(fashion_mnist)
                ntot, dx, dy = X.shape
                X = X.reshape(ntot, -1) ; ntot, d = X.shape
                F = get_F((p,d), typeF='dct')
                scaler = preprocessing.StandardScaler().fit(X)
                X = scaler.transform(X)
            elif which_transform == 'random_gaussian_features':
                if which_non_lin == 'erf':
                    fashion_mnist = datasets.FashionMNIST(root='data', train=True, download=True, transform=None)
                    X = preprocess_data_fashionmnist(fashion_mnist)
                    ntot, dx, dy = X.shape
                    X = X.reshape(ntot, -1)
                    ntot, d = X.shape
                    F = np.random.normal(0., 1., size = (d, p))
                    X = special.erf(X @ F)
                elif which_non_lin == 'tanh':
                    fashion_mnist = datasets.FashionMNIST(root='data', train=True, download=True, transform=None)
                    X = preprocess_data_fashionmnist(fashion_mnist)
                    ntot, dx, dy = X.shape
                    X = X.reshape(ntot, -1)
                    ntot, d = X.shape
                    F = np.random.normal(0., 1., size = (d, p))
                    X = np.tanh(X @ F)
                elif which_non_lin == 'sign':
                    fashion_mnist = datasets.FashionMNIST(root='data', train=True, download=True, transform=None)
                    X = preprocess_data_fashionmnist(fashion_mnist)
                    ntot, dx, dy = X.shape
                    X = X.reshape(ntot, -1)
                    ntot, d = X.shape
                    F = np.random.normal(0., 1., size = (d, p))
                    X = np.sign(X @ F)
                elif which_non_lin == 'relu':
                    fashion_mnist = datasets.FashionMNIST(root='data', train=True, download=True, transform=None)
                    X = preprocess_data_fashionmnist(fashion_mnist)
                    ntot, dx, dy = X.shape
                    X = X.reshape(ntot, -1)
                    ntot, d = X.shape
                    F = np.random.normal(0., 1., size = (d, p))
                    X = relu(X @ F)
        elif which_real_dataset == "CIFAR10":
            if which_transform == 'no_transform':
                transform = transforms.Compose([transforms.Grayscale(num_output_channels = 1), transforms.ToTensor()])
                cifar10 = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
                X = preprocess_data_cifar10(cifar10)
                ntot, dx, dy = X.shape
                X = X.reshape(ntot, -1)
            elif which_transform == 'other':
                transform = transforms.Compose([transforms.Grayscale(num_output_channels = 1), transforms.ToTensor()])
                cifar10 = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
                X = preprocess_data_cifar10(cifar10)
                ntot, dx, dy = X.shape
                X = X.reshape(ntot, -1) ; ntot, d = X.shape
                F = get_F((p,d), typeF='orthogonal')
                scaler = preprocessing.StandardScaler().fit(X)
                X = scaler.transform(X)
            elif which_transform == 'random_gaussian_features':
                if which_non_lin == 'erf':
                    transform = transforms.Compose([transforms.Grayscale(num_output_channels = 1), transforms.ToTensor()])
                    cifar10 = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
                    X = preprocess_data_cifar10(cifar10)
                    ntot, dx, dy = X.shape
                    X = X.reshape(ntot, -1)
                    ntot, d = X.shape
                    F = np.random.normal(0., 1., size = (d, p))
                    X = special.erf(X @ F)
                elif which_non_lin == 'tanh':
                    transform = transforms.Compose([transforms.Grayscale(num_output_channels = 1), transforms.ToTensor()])
                    cifar10 = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
                    X = preprocess_data_cifar10(cifar10)
                    ntot, dx, dy = X.shape
                    X = X.reshape(ntot, -1)
                    ntot, d = X.shape
                    F = np.random.normal(0., 1., size = (d, p))
                    X = np.tanh(X @ F)
                elif which_non_lin == 'sign':
                    transform = transforms.Compose([transforms.Grayscale(num_output_channels = 1), transforms.ToTensor()])
                    cifar10 = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
                    X = preprocess_data_cifar10(cifar10)
                    ntot, dx, dy = X.shape
                    X = X.reshape(ntot, -1)
                    ntot, d = X.shape
                    F = np.random.normal(0., 1., size = (d, p))
                    X = np.sign(X @ F)
                elif which_non_lin == 'relu':
                    transform = transforms.Compose([transforms.Grayscale(num_output_channels = 1), transforms.ToTensor()])
                    cifar10 = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
                    X = preprocess_data_cifar10(cifar10)
                    ntot, dx, dy = X.shape
                    X = X.reshape(ntot, -1)
                    ntot, d = X.shape
                    F = np.random.normal(0., 1., size = (d, p))
                    X = relu(X @ F)
        hf = h5py.File(path_to_data_folder + '/' + 'X_%s_%s_%s.hdf5'%(which_real_dataset, which_transform, which_non_lin), 'w')
        hf.create_dataset('X_%s_%s_%s'%(which_real_dataset, which_transform, which_non_lin), data=X)
        hf.close()
    return