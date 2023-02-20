import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.special import erf
from scipy.integrate import quad
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LogisticRegression
from scipy.optimize import minimize_scalar
from sklearn.preprocessing import normalize
from math import factorial
from sklearn.preprocessing import normalize
from scipy.integrate import quad



## Generation and ERM train for Gaussian and GMM distribution

def ridge_estimator(X, y, lamb=0.1):
    '''
    Implements the pseudo-inverse ridge estimator.
    '''
    m, n = X.shape
    if m >= n:
        return np.linalg.inv(X.T @ X + lamb*np.identity(n)) @ X.T @ y
    elif m < n:
        return X.T @ np.linalg.inv(X @ X.T + lamb*np.identity(m)) @ y
def get_data_gmm(n,theta,noise,gmm_params):
    mean_plus , mean_minus , Tplus, Tminus , p1 = gmm_params
    d = len(theta)
    nplus = int(p1*n) ; nminus = int(n-nplus)
    Xplus = mean_plus/np.sqrt(d)*np.ones((nplus,d)) + np.random.normal(0,1,(nplus,d))@Tplus
    Xminus = mean_minus/np.sqrt(d)*np.ones((nminus,d)) + np.random.normal(0,1,(nminus,d))@Tminus
    X = np.vstack((Xplus,Xminus))
    y = X @ theta/(np.sqrt(d)) + np.sqrt(noise)*np.random.randn(n)
    return X/np.sqrt(d), y


def get_errors_gmm(n,theta,noise,gmm_params,flag,lamb,seeds=10):
    d = len(theta)
    eg, et = [], [] ; ws = np.zeros((int(seeds),int(d)))
    # Iterate over different realisations of the problem.
    ntest = 2000
    for i in range(seeds):
        if flag:
            print('Seed: {}'.format(i))
        Xp_train, yp_train = get_data_gmm(n,theta,noise,gmm_params)
        Xp_test, yp_test = get_data_gmm(ntest,theta,noise,gmm_params)
        w = ridge_estimator(Xp_train, yp_train, lamb)
        yhat_train = Xp_train @ w
        yhat_test = Xp_test @ w
        # Train loss
        train_loss = np.mean((yp_train - yhat_train)**2)
        # Fresh samples
        test_error = np.mean((yp_test - yhat_test)**2) 
        eg.append(test_error)   ;  et.append(train_loss)
        ws[i,:] = w
    # Return average and standard deviation of both errors
    return (np.mean(et), np.std(et)/(np.sqrt(seeds)), 
            np.mean(eg), np.std(eg)/(np.sqrt(seeds)), ws.mean(axis=0)) 
def simulate_gmm(alphas,theta,noise,gmm_params, flag, lamb, seeds = 10):
    data = {'test_error': [], 'train_loss': [], 'test_error_std': [],
            'train_loss_std': [], 'lambda': [], 'sample_complexity': [], 'what':[],'hsimul':[]}
    for alpha in alphas:
        d = len(theta)
        n = int(d*alpha)
        if flag:
            print('Simulating sample complexity: {}'.format(alpha))
        et, et_std, eg, eg_std, w = get_errors_gmm(n,theta,noise,gmm_params,flag, lamb,seeds=seeds)
        print(f'training error for alpha = {alpha} is {et}')
        h = np.dot(w,np.ones(d))/d
        print(f'overlap estimator with mean is {h}')
        data['sample_complexity'].append(alpha) ; data['hsimul'].append(h)
        data['what'].append(w)
        data['lambda'].append(lamb)
        data['test_error'].append(eg)
        data['test_error_std'].append(eg_std)
        data['train_loss'].append(et)
        data['train_loss_std'].append(et_std)

    return pd.DataFrame.from_dict(data)
def get_data_gem(n,theta,noise,gem_params):
    mean_gcm , T = gem_params
    d = len(theta)
    X = mean_gcm/np.sqrt(d)*np.ones((n,d)) + np.random.normal(0,1,(n,d))@T
    y = X @ theta/(np.sqrt(d)) + np.sqrt(noise)*np.random.randn(n)
    return X/np.sqrt(d), y

def get_errors_gem(n,theta,noise,gem_params,flag,lamb,seeds=10):
    d = len(theta)
    eg, et = [], [] ; ws = np.zeros((int(seeds),int(d)))
    # Iterate over different realisations of the problem.
    ntest = 2000
    for i in range(seeds):
        if flag:
            print('Seed: {}'.format(i))
        Xp_train, yp_train = get_data_gem(n,theta,noise,gem_params)
        Xp_test, yp_test = get_data_gem(ntest,theta,noise,gem_params)
        w = ridge_estimator(Xp_train, yp_train, lamb)
        yhat_train = Xp_train @ w
        yhat_test = Xp_test @ w
        # Train loss
        train_loss = np.mean((yp_train - yhat_train)**2)
        # Fresh samples
        test_error = np.mean((yp_test - yhat_test)**2) 
        eg.append(test_error)   ;  et.append(train_loss)
        ws[i,:] = w
    # Return average and standard deviation of both errors
    return (np.mean(et), np.std(et)/(np.sqrt(seeds)), 
            np.mean(eg), np.std(eg)/(np.sqrt(seeds)), ws.mean(axis=0)) 
def simulate_gem(alphas,theta,noise,gem_params, flag, lamb, seeds = 10):
    data = {'test_error': [], 'train_loss': [], 'test_error_std': [],
            'train_loss_std': [], 'lambda': [], 'sample_complexity': [], 'what':[],'hsimul':[]}
    d = len(theta)
    for alpha in alphas:
        n = int(d*alpha)
        if flag:
            print('Simulating sample complexity: {}'.format(alpha))
        et, et_std, eg, eg_std, w = get_errors_gem(n,theta,noise,gem_params,flag, lamb,seeds=seeds)
        print(f'training error for alpha = {alpha} is {et}')
        h = np.dot(w,np.ones(d))/d
        print(f'overlap estimator with mean is {h}')
        data['sample_complexity'].append(alpha) ; data['hsimul'].append(h)
        data['what'].append(w)
        data['lambda'].append(lamb)
        data['test_error'].append(eg)
        data['test_error_std'].append(eg_std)
        data['train_loss'].append(et)
        data['train_loss_std'].append(et_std)

    return pd.DataFrame.from_dict(data)
