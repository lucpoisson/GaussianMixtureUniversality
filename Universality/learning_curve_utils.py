import os
import numpy as np
import pandas as pd
import csv
import h5py
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
import cvxpy as cp
from sklearn.model_selection import train_test_split

def load_data(which_dataset, which_transform, which_non_lin):
    filename = "./data/X_%s_%s_%s.hdf5"%(which_dataset, which_transform, which_non_lin)
    with h5py.File(filename, "r") as f:
        # List all groups
        print("Keys: %s" % f.keys())
        a_group_key = list(f.keys())[0]
        # Get the data
        X = np.asarray(list(f[a_group_key]))
    M, N = X.shape
    ## Forget the real label
    y = np.zeros(M)
    return X, y
def ridge_estimator(X, y, lamb=0.1):
    m, n = X.shape
    if m >= n:
        return np.linalg.inv(X.T @ X + lamb*np.identity(n)) @ X.T @ y
    elif m < n:
        return X.T @ np.linalg.inv(X @ X.T + lamb*np.identity(m)) @ y
def get_estimator(X_train, y_train, λ, loss = "square_loss", solver = 'cvxpy'):
    n, p = X_train.shape
    if loss == "square_loss":
        W = ridge_estimator(X_train/np.sqrt(p), y_train,lamb=λ)
    elif loss == 'logistic_loss':
        if solver == 'cvxpy':
            W = cp.Variable((p))
            l = cp.sum(cp.logistic(cp.multiply(y_train, (X_train @ W/np.sqrt(p)))))
            reg = (cp.norm(W, 2))**2
            lambd = cp.Parameter(nonneg=True)
            prob = cp.Problem(cp.Minimize(l + lambd*reg))
            lambd.value = λ
            prob.solve()
            W = W.value
        else:
            W = LogisticRegression(penalty='l2',solver='lbfgs',fit_intercept=False, C = λ**(-1),
                               max_iter=1e10, tol=1e-3, verbose=0).fit(X_train/np.sqrt(p),y_train).coef_[0]
    elif loss == 'hinge_loss':
        if solver == 'cvxpy':
            W = cp.Variable((p))
            l = cp.sum(cp.pos(1 - cp.multiply(y_train, (X_train @ W/np.sqrt(p)))))
            reg = (cp.norm(W, 2))**2
            lambd = cp.Parameter(nonneg=True)
            prob = cp.Problem(cp.Minimize(l + lambd*reg))
            lambd.value = λ
            prob.solve()
            W = W.value
        elif solver == 'sk':
            tol = 1e-5
            maxiter = 10000
            W = LinearSVC(penalty='l2', loss='hinge', dual=True, tol=tol, C=λ**(-1), multi_class='ovr',fit_intercept=False, intercept_scaling=0.0, class_weight=None,
                          verbose=False, random_state=None, max_iter=maxiter).fit(X_train/np.sqrt(p), y_train).coef_[0]
    return W
def mse(y,yhat):
    return np.mean((y-yhat)**2)
def logistic_loss(z):
    return np.log(1+np.exp(-z))
def hinge_loss(z):
    return np.maximum(np.zeros(len(z)), np.ones(len(z)) - z)

def get_train_loss(X_train, y_train, W, loss = "square"):
    n, p = X_train.shape
    if loss == "square_loss":
        train_loss = 0.5*mse(y_train, (X_train @ W)/np.sqrt(p))
    elif loss == "logistic_loss":
        train_loss = np.mean(logistic_loss((X_train @ W)/np.sqrt(p) * y_train))
    elif loss == "hinge_loss":
        train_loss = np.mean(hinge_loss((X_train @ W)/np.sqrt(p) * y_train))
    return train_loss

def get_test_error(X_test, y_test, W, loss='square'):
    n, p = X_test.shape
    if loss == "square_loss":
        yhat_test = X_test @ W/(np.sqrt(p))
        test_error = np.mean((y_test - yhat_test)**2)  
    elif loss == "logistic_loss":
        yhat_test = np.sign(X_test @ W)
        test_error = 0.25*np.mean((y_test - yhat_test)**2)
    return test_error

def get_learning_curve_real(Cov,noise,αs = [0.1], λ = 1e-15, task = 'classification' , loss = 'logistic', which_real_dataset = 'MNIST', which_transform = 'wavelet_scattering', which_non_lin = 'erf',
                        path_to_data_folder = './', path_to_res_folder = './', solver = 'cvxpy', seed = 1):

    sim = {'alpha': [], 'train_loss': [], 'test_error':[] ,'p': [], 'which_real_dataset': [], 'which_transform': [], 'which_non_lin': [],
           'loss': [], 'lamb': [], 'q':[],'m':[]}

    X , y = load_data(which_real_dataset, which_transform, which_non_lin)
    n, p = X.shape
    for α in αs:
        n = int(np.floor(α * p))
        X_train, X_test, y_train, y_test = train_test_split(X, y, train_size = n , test_size = 10000 , random_state=42)
        teacher = np.random.randn(p) 
        if task == 'classification':
            y_train = np.sign(X_train@teacher/(np.sqrt(p)) + np.sqrt(noise)*np.random.randn(n)) 
            y_test = np.sign(X_test@teacher/(np.sqrt(p)) + np.sqrt(noise)*np.random.randn(10000)) 
        else: 
            y_train = X_train@teacher/(np.sqrt(p)) + np.sqrt(noise)*np.random.randn(n)
            y_test = X_test@teacher/(np.sqrt(p)) + np.sqrt(noise)*np.random.randn(10000)
        n, p = X_train.shape
        W = get_estimator(X_train, y_train, λ, loss = loss, solver = solver)
        train_loss = get_train_loss(X_train, y_train, W, loss = loss)
        test_error = get_test_error(X_test,y_test,W,loss=loss)
        ovq = np.dot(W,Cov@W)/(p) ; ovm = np.dot(W,Cov@teacher)/(p)
        sim['p'].append(p)
        sim['alpha'].append(α)
        sim['which_real_dataset'].append(which_real_dataset)
        sim['which_transform'].append(which_transform)
        sim['which_non_lin'].append(which_non_lin)
        sim['loss'].append(loss)
        sim['lamb'].append(str(λ))
        sim['train_loss'].append(train_loss)
        sim['test_error'].append(test_error)
        sim['q'].append(ovq) ; sim['m'].append(ovm)
    if os.path.isdir(path_to_res_folder) == False: # create the results folder if not there
        try:
            os.makedirs(path_to_res_folder)
        except OSError:
            print ("\n!!! ERROR: Creation of the directory %s failed" % path_to_res_folder)
            raise

    if os.path.isdir(path_to_res_folder + "/seeds") == False: # create the seed folder if not there
        try:
            os.makedirs(path_to_res_folder  + "/seeds")
        except OSError:
            print ("\n!!! ERROR: Creation of the directory %s failed" % path_to_res_folder)
            raise

    sim = pd.DataFrame.from_dict(sim)
    if os.path.isfile(path_to_res_folder  + "/seeds" + "/sim_%s_seed_%d_real.csv"%(loss,seed)) == False:
        with open(path_to_res_folder + "/seeds" + "/sim_%s_seed_%d_real.csv"%(loss,seed), mode='w') as f:
            wr = csv.writer(f, dialect='excel')
            wr.writerow(sim.keys().to_list())
    sim.to_csv(path_to_res_folder + "/seeds/" + "/sim_%s_seed_%d_real.csv"%(loss,seed), mode = 'a', index = False, header = None)
    return

def build_covariances_real(p, which_real_dataset = "none", which_transform = "none",
                    which_non_lin = "none", path_to_data_folder = "./"):

    μ = np.zeros(p)
    filename = path_to_data_folder + "/X_%s_%s_%s.hdf5"%(which_real_dataset, which_transform, which_non_lin)
    with h5py.File(filename, "r") as f:
        # List all groups
        print("Keys: %s" % f.keys())
        a_group_key = list(f.keys())[0]
        # Get the data
        X = np.asarray(list(f[a_group_key]))
        M_tot, N = X.shape
        Σ = (X.T @ X)/M_tot # compute the empirical covariance matrix
    return (μ, Σ)



def statistics_learning_curve_real(n_seeds = 10, loss = 'square_loss', path_to_res_folder = './results'):

    res = {'p': [], 'alpha': [], 'which_real_dataset': [],
              'which_transform': [], 'which_non_lin': [],
              'loss': [], 'lamb': [], 'train_loss':[], 'test_error':[],'m':[],'q':[]}

    for s in range(n_seeds):
        df = pd.read_csv(path_to_res_folder + "/seeds/sim_%s_seed_%d_real.csv"%(loss,s))
        res['alpha'].append(df['alpha'])
        res['train_loss'].append(df['train_loss'])
        res['test_error'].append(df['test_error'])
        res['m'].append(df['m'])
        res['q'].append(df['q'])
        res['p'].append(df['p'])
        res['which_real_dataset'].append(df['which_real_dataset'])
        res['which_transform'].append(df['which_transform'])
        res['which_non_lin'].append(df['which_non_lin'])
        res['loss'].append(df['loss'])
        res['lamb'].append(df['lamb'])
    res_tmp = res.copy()
    res_tmp =  pd.DataFrame.from_dict(res_tmp)
    res_tmp.which_real_dataset = res_tmp.which_real_dataset.apply(str)
    res_tmp.which_transform = res_tmp.which_transform.apply(str)
    res_tmp.which_non_lin = res_tmp.which_non_lin.apply(str)
    res_tmp.lamb = res_tmp.lamb.apply(str)
    res_stat = res_tmp.groupby(['which_real_dataset', 'which_transform', 'which_non_lin', 'lamb']).aggregate(mean_train_loss = ('train_loss', lambda x: np.vstack(x).mean(axis=0).tolist()), mean_test_error = ('test_error', lambda x: np.vstack(x).mean(axis=0).tolist()),
                                                std_train_loss = ('train_loss', lambda x: np.vstack(x).std(axis=0).tolist()),  std_test_error = ('test_error', lambda x: np.vstack(x).std(axis=0).tolist()) , 
                                                mean_q = ('q', lambda x: np.vstack(x).mean(axis=0).tolist()), mean_m = ('m', lambda x: np.vstack(x).mean(axis=0).tolist()),
                                                std_q = ('q', lambda x: np.vstack(x).std(axis=0).tolist()),  std_m = ('m', lambda x: np.vstack(x).std(axis=0).tolist()) 
                                                ).reset_index()

    if os.path.isfile(path_to_res_folder + "/sim_%s_real.csv"%(loss)) == False:
        with open(path_to_res_folder + "/sim_%s_real.csv"%(loss), mode='w') as f:
            f.write("alpha,mean_train_loss,std_train_loss,mean_test_error,std_test_error,mean_q,std_q,mean_m,std_m,p,n_seeds,which_real_dataset,which_transform,which_which_non_lin,loss,lamb\n")

    with open(path_to_res_folder + "/sim_%s_real.csv"%(loss), mode='a') as f:
        wr = csv.writer(f, dialect='excel')
        for i in range(len(res_stat.index)):
            j = 0
            n_seeds = len(res['alpha'])
            for alpha in res['alpha'][0]:
                wr.writerow([alpha, 
                res_stat['mean_train_loss'][i][j],res_stat['std_train_loss'][i][j]/np.sqrt(n_seeds),
                             res_stat['mean_test_error'][i][j] , res_stat['std_test_error'][i][j]/np.sqrt(n_seeds),
                res_stat['mean_q'][i][j],res_stat['std_q'][i][j]/np.sqrt(n_seeds),
                             res_stat['mean_m'][i][j] , res_stat['std_m'][i][j]/np.sqrt(n_seeds),
                n_seeds, res['p'][i][j], res['which_real_dataset'][i][j], res['which_transform'][i][j], res['which_non_lin'][i][j],
                res['loss'][i][j], res['lamb'][i][j]])
                j += 1
    return