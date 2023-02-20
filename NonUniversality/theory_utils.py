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


### Gaussian asymptotic theory curves 

def replica_ridge(alphas,Omega,Phi,Psi,theta,lamb,omega,Delta,mean,mean0,noise=0,flag=True):   
    data_model = Custom(teacher_teacher_cov=Psi,
                     student_student_cov=Omega,
                     teacher_student_cov=Phi,
                     teacher=theta.flatten(),
                     fixed_teacher=True,
                     mean=mean , mean0=mean0,noise=noise)
    experiment = CustomExperiment(task = 'ridge_regression', 
                                  regularisation = lamb, 
                                  data_model = data_model, 
                                  tolerance = 1e-12, 
                                  damping = 0.5, 
                                  verbose = flag, 
                                  max_steps = 2000,
                                 omega=omega,
                                 Delt=Delta)
    experiment.learning_curve(alphas =[alphas[0]])
    replicas = experiment.get_curve()
    for i in range(1,len(alphas)):
        experiment.learning_curve(alphas =[alphas[i]])
        replicas = pd.concat([replicas,experiment.get_curve()])
    return replicas

#### Data Model 

class DataModel(object):
    '''
    Base data model.
    '''
    def __init__(self):
        self.p, self.d = self.Phi.shape

        self._diagonalise()
        self._check_commute()

    def get_info(self):
        info = {
            'data_model': 'base',
            'latent_dimension': self.d,
            'input_dimension': self.p
        }
        return info

    def _check_commute(self):
        if False:#np.linalg.norm(self.Omega @ self.PhiPhiT - self.PhiPhiT @ self.Omega) < 1e-10:
            self.commute = True
        else:
            self.commute = False
            #self._UTPhiPhiTU = self.eigv_Omega.T @ self.PhiPhiT @ self.eigv_Omega
            self._UTPhiPhiTU = np.diagonal(self.eigv_Omega.T @ self.PhiPhiT @ self.eigv_Omega)
            self._UTmumuTU = np.diagonal(self.eigv_Omega.T @ self.mumuT @ self.eigv_Omega)
            self._UTmuthetaTPhiTU = np.diagonal(self.eigv_Omega.T @ self.muthetaTPhi @ self.eigv_Omega)

    def _diagonalise(self):
        '''
        Diagonalise covariance matrices.
        '''
        self.spec_Omega, self.eigv_Omega = np.linalg.eigh(self.Omega)
        self.spec_Omega = np.real(self.spec_Omega)

        self.spec_PhiPhit = np.real(np.linalg.eigvalsh(self.PhiPhiT))
        self.spec_mumuT = np.real(np.linalg.eigvalsh(self.mumuT))
        self.spec_muthetaTPhi = np.real(np.linalg.eigvalsh(self.muthetaTPhi))
class Model(object):
    '''
    Base class for a model.
    -- args --
    sample_complexity: sample complexity
    regularisation: ridge penalty coefficient
    data_model: data_model instance. See /data_model/
    '''
    def __init__(self, *, sample_complexity, regularisation, data_model):
        self.alpha = sample_complexity
        self.lamb = regularisation
        self.data = data_model

        self.parameters, self.dimension = self.data_model.Phi.shape
        self.gamma = self.dimension / self.parameters

    def get_info(self):
        '''
        Information about the model.
        '''
        info = {
            'model': 'generic',
            'sample_complexity': self.alpha,
            'lambda': self.lamb,
        }
        return info

    def update_se(self, V, q, m):
        '''
        Method for t -> t+1 update in saddle-point iteration.
        '''
        raise NotImplementedError

    def get_test_error(self, q, m):
        '''
        Method for computing the test error from overlaps.
        '''
        raise NotImplementedError

    def get_train_loss(self, V, q, m):
        '''
        Method for computing the training loss from overlaps.
        '''
        raise NotImplementedError


#### Custom 

class Custom(DataModel):
    '''
    Custom allows for user to pass his/her own covariance matrices.
    -- args --
    teacher_teacher_cov: teacher-teacher covariance matrix (Psi)
    student_student_cov: student-student covariance matrix (Omega)
    teacher_student_cov: teacher-student covariance matrix (Phi)
    '''
    def __init__(self, *, teacher_teacher_cov, student_student_cov, teacher_student_cov,
                 teacher, fixed_teacher, mean , mean0,noise=0):
        self.Psi = teacher_teacher_cov
        self.Omega = student_student_cov
        self.Phi = teacher_student_cov
        self.teacher=teacher
        self.fixed_teacher=fixed_teacher
        self.mean=mean
        self.mean0=mean0
        
        
        self.p, self.k = self.Phi.shape
        self.gamma = self.k / self.p
        if not self.fixed_teacher:
            self.PhiPhiT = self.Phi @ self.Phi.T
            self.rho = np.trace(self.Psi) / self.k + noise
        if self.fixed_teacher:
            self.PhiPhiT = self.Phi @(self.teacher.reshape(-1,1)@self.teacher.reshape(1,-1))@self.Phi.T
            self.rho = self.teacher@(self.Psi@self.teacher) / self.k + noise
        self.mumuT=self.mean.reshape(-1,1)@self.mean.reshape(1,-1)
        self.muthetaTPhi=self.mean.reshape(-1,1)@self.teacher.reshape(1,-1)@self.Phi.T
        self.h0 = self.teacher.flatten()@self.mean0.flatten()/self.k
        self._check_sym()
        self._diagonalise() # see base_data_model
        self._check_commute()

    def get_info(self):
        info = {
            'data_model': 'custom',
            'teacher_dimension': self.k,
            'student_dimension': self.p,
            'aspect_ratio': self.gamma,
            'rho': self.rho
        }
        return info

    def _check_sym(self):
        '''
        Check if input-input covariance is a symmetric matrix.
        '''
        if (np.linalg.norm(self.Omega - self.Omega.T) > 1e-5):
            print('Student-Student covariance is not a symmetric matrix. Symmetrizing!')
            self.Omega = .5 * (self.Omega+self.Omega.T)

        if (np.linalg.norm(self.Psi - self.Psi.T) > 1e-5):
            print('Teacher-teaccher covariance is not a symmetric matrix. Symmetrizing!')
            self.Psi = .5 * (self.Psi+self.Psi.T)


class CustomSpectra(DataModel):
    '''
    Custom allows for user to pass directly the spectra of the covarinces.
    -- args --
    spec_Psi: teacher-teacher covariance matrix (Psi)
    spec_Omega: student-student covariance matrix (Omega)
    spec_UPhiPhitUT: projection of student-teacher covariance into basis of Omega
    '''
    def __init__(self, *, rho, spec_Omega, spec_UPhiPhitUT, gamma):
        self.rho = rho
        self.spec_Omega = spec_Omega
        self._UTPhiPhiTU = spec_UPhiPhitUT

        self.p = len(self.spec_Omega)
        self.gamma = gamma
        self.k = int(self.gamma * self.p)

        self.commute = False

    def get_info(self):
        info = {
            'data_model': 'custom_spectra',
            'teacher_dimension': self.k,
            'student_dimension': self.p,
            'aspect_ratio': self.gamma,
            'rho': self.rho
        }
        return info


#### Custom experiment 

class CustomExperiment(object):
    '''
    Implements experiment for generic task and custom, for fixed
    regularisation.

    Note sample complexity is passed to run_experiment as an argument
    allowing for running several sample complexities for the same pre-diagonalised
    data model.
    '''
    def __init__(self,initialisation='uninformed', tolerance=1e-10, damping=0,
                       verbose=False, max_steps=1000,omega=0, Delt=1,*,
                       task, regularisation, data_model):

        self.task = task
        self.lamb = regularisation
        self.data_model = data_model
        self.omega=omega
        self.Delt=Delt

        # Hyperparameters
        self.initialisation=initialisation
        self.tolerance = tolerance
        self.damping = damping
        self.verbose = verbose
        self.max_steps = max_steps


    def learning_curve(self, *, alphas):
        curve = {
            'task': [],
            'gamma': [],
            'lambda': [],
            'rho': [],
            'h0':[],
            'sample_complexity': [],
            'V': [],
            'm': [],
            'q': [],
            'h':[],
            "Vhat":[],
            "mhat":[],
            "qhat":[],
            'hhat':[],
            'test_error': [],
            'train_loss': [],
        }
        for alpha in alphas:
            if self.verbose:
                print('Runninig sample complexity: {}'.format(alpha))

            self._run(sample_complexity = alpha)
            info_sp = self.se.get_info()
            info_data = self.data_model.get_info()

            curve['task'].append(self.task)
            curve['gamma'].append(info_data['teacher_dimension'] / info_data['student_dimension'])
            curve['lambda'].append(self.lamb)
            curve['rho'].append(self.data_model.rho)
            curve['h0'].append(self.data_model.h0)
            curve['sample_complexity'].append(alpha)

            curve['test_error'].append(info_sp['test_error'])
            curve['train_loss'].append(info_sp['train_loss'])
            curve['V'].append(info_sp['overlaps']['variance'])
            curve['q'].append(info_sp['overlaps']['self_overlap'])
            curve['m'].append(info_sp['overlaps']['teacher_student'])
            curve['h'].append(info_sp['overlaps']['student_mean'])
            curve['Vhat'].append(info_sp['hatoverlaps']['variance'])
            curve['qhat'].append(info_sp['hatoverlaps']['self_overlap'])
            curve['mhat'].append(info_sp['hatoverlaps']['teacher_student'])
            curve['hhat'].append(info_sp['hatoverlaps']['student_mean'])

        self._learning_curve = pd.DataFrame.from_dict(curve)


    def get_curve(self):
        return self._learning_curve

    def _run(self, *, sample_complexity):
        '''
        Runs saddle-point equations.
        '''
        self._initialise_model(sample_complexity)

        self.se = StateEvolution(model=self.model,
                       initialisation=self.initialisation,
                       tolerance=self.tolerance,
                       damping=self.damping,
                       verbose=True,
                       max_steps=self.max_steps)

        self.se.iterate()

    def _initialise_model(self, sample_complexity):
        if self.task == 'ridge_regression':
            self.model = RidgeRegression(sample_complexity = sample_complexity,
                                         regularisation=self.lamb,
                                         data_model = self.data_model,
                                        om = self.omega,
                                        Delt = self.Delt
                                        )


        elif self.task == 'logistic_regression':
            self.model = LogisticRegression(sample_complexity = sample_complexity,
                                            regularisation=self.lamb,
                                            data_model = self.data_model,
                                           om=self.omega,
                                        Delt=self.Delt)
        else:
            print('{} not implemented.'.format(self.task))

    def save_experiment(self, date=False, unique_id=False, directory='.', *, name):
        '''
        Saves result of experiment in .json file with info for reproductibility.
        '''
        path = '{}/{}'.format(directory, name)

        if date:
            from datetime import datetime
            day, time = datetime.now().strftime("%d_%m_%Y"), datetime.now().strftime("%H:%M")
            path += '_{}_{}'.format(day, time)

        if unique_id:
            import uuid
            unique_id = uuid.uuid4().hex
            path += '_{}'.format(unique_id)

        path += '.csv'
        print('Saving experiment at {}'.format(path))
        self._learning_curve.to_csv(path, index=False)

#### Ridge

class RidgeRegression(Model):
    '''
    Implements updates for ridge regression task.
    See base_model for details on modules.
    '''
    def __init__(self, *, sample_complexity, regularisation, data_model,om,Delt):
        self.alpha = sample_complexity
        self.lamb = regularisation

        self.data_model = data_model
        self.om=om
        self.Delt=Delt

    def get_info(self):
        info = {
            'model': 'ridge_regression',
            'sample_complexity': self.alpha,
            'lambda': self.lamb,
        }
        return info

    def _update_overlaps(self, Vhat, qhat, mhat,hhat):
        V = np.mean(self.data_model.spec_Omega/(self.lamb + Vhat * self.data_model.spec_Omega))

        if self.data_model.commute:
            q = np.mean(self.data_model.spec_Omega*(self.data_model.spec_Omega * qhat +
                                               mhat**2 *  self.data_model.spec_PhiPhit+2*hhat*mhat*self.data_model.spec_muthetaTPhi
                                                    +hhat**2*self.data_model.spec_mumuT) /
                                          (self.lamb + Vhat*self.data_model.spec_Omega)**2)

            m = 1/np.sqrt(self.data_model.gamma) * np.mean((mhat*self.data_model.spec_PhiPhit+
                                                            hhat*self.data_model.spec_muthetaTPhi)/
                                                           (self.lamb + Vhat*self.data_model.spec_Omega))

            h=np.mean((hhat*self.data_model.spec_mumuT+
                       mhat*self.data_model.spec_muthetaTPhi)/(self.lamb + Vhat * self.data_model.spec_Omega))

        else:
            q = qhat * np.mean(self.data_model.spec_Omega**2 / (self.lamb + Vhat*self.data_model.spec_Omega)**2)
            q += np.mean((mhat**2 *  self.data_model._UTPhiPhiTU+2*hhat*mhat*self.data_model._UTmuthetaTPhiTU
                                                    +hhat**2*self.data_model._UTmumuTU)
                         * self.data_model.spec_Omega/(self.lamb + Vhat * self.data_model.spec_Omega)**2)

            m = 1/np.sqrt(self.data_model.gamma) * np.mean((mhat*self.data_model._UTPhiPhiTU+hhat*self.data_model._UTmuthetaTPhiTU)
                                                           /(self.lamb + Vhat * self.data_model.spec_Omega))

            h=np.mean((hhat*self.data_model._UTmumuTU+mhat*self.data_model._UTmuthetaTPhiTU)/(self.lamb + Vhat * self.data_model.spec_Omega))
        return V, q, m, h

    def _update_hatoverlaps(self, V, q, m, h):
        if self.lamb>1e-6:
            #print("rescaled")
            Vhat = self.alpha * 1/(1+V)
            qhat = self.alpha * (self.data_model.rho + q - 2*m + (h - self.data_model.h0)**2)/(1+V)**2
            mhat = self.alpha/np.sqrt(self.data_model.gamma) * 1/(1+V)
            hhat = self.alpha*(self.data_model.h0-h)/(1+V)
        else:
            print('Increase regularization at the moment, we will fix it.')
        return Vhat, qhat, mhat,hhat
    def update_se(self, V, q, m, h):
        Vhat, qhat, mhat, hhat  = self._update_hatoverlaps(V, q, m, h)
        return self._update_overlaps(Vhat, qhat, mhat, hhat)
    def get_test_error(self, q, m, h):
        return self.data_model.rho + q + (h - self.data_model.h0)**2 - 2*m
    def get_train_loss(self, V, q, m, h):
        return (self.data_model.rho + q + (h - self.data_model.h0)**2 - 2*m) / (1+V)**2

#### SE 

class StateEvolution(object):
    '''
    Iterator for the saddle-point equations.
    -- args --
    initialisation: initial condition (uninformed or informed)
    tolerante: tolerance for convergence.
    damping: damping coefficient.
    verbose: if true, print step-by-step iteration.
    max_steps: maximum number of steps before convergence
    model: instance of model class. See /models.
    '''
    def __init__(self, initialisation=None, tolerance=1e-10, damping=0,
                 verbose=True, max_steps=1000, *, model):

        self.max_steps = max_steps
        self.init = initialisation
        self.tol = tolerance
        self.damping = damping
        self.model = model
        self.verbose = verbose

        # Status = 0 at initialisation.
        self.status = 0


    def _initialise(self):
        '''
        Initialise saddle-point equations
        '''
        self.overlaps = {
            'variance': np.zeros(self.max_steps+1),
            'self_overlap': np.zeros(self.max_steps+1),
            'teacher_student': np.zeros(self.max_steps+1),
            'student_mean': np.zeros(self.max_steps+1)
            
        }
        
        self.hatoverlaps = {
            'variance': np.zeros(self.max_steps+1),
            'self_overlap': np.zeros(self.max_steps+1),
            'teacher_student': np.zeros(self.max_steps+1),
            'student_mean': np.zeros(self.max_steps+1)
        }
        
        if not isinstance(self.init, str):
            self.overlaps['variance'][0] = self.init[0]
            self.overlaps['self_overlap'][0] = self.init[1]
            self.overlaps['teacher_student'][0] = self.init[2]
            self.overlaps['student_mean'][0] = self.init[3]
        
        elif self.init == 'uninformed':
            self.overlaps['variance'][0] = 1000
            self.overlaps['self_overlap'][0] = 0.001
            self.overlaps['teacher_student'][0] = 0.001
            self.overlaps['student_mean'][0] = 0.001
            
            
            

        elif self.init == 'informed':
            self.overlaps['variance'][0] = 0.001
            self.overlaps['self_overlap'][0] = 0.999
            self.overlaps['teacher_student'][0] = 0.999
            self.overlaps['student_mean'][0] = 0.999
        
            
        self.hatoverlaps['variance'][0],self.hatoverlaps['self_overlap'][0],self.hatoverlaps['teacher_student'][0],self.hatoverlaps['student_mean'][0]=self.model._update_hatoverlaps(
                self.overlaps['variance'][0],self.overlaps['self_overlap'][0],
                                               self.overlaps['teacher_student'][0],   self.overlaps['student_mean'][0])


    def _get_diff(self, t):
        '''
        Compute differencial between step t+1 and t.
        '''
        diff = np.abs(self.overlaps['variance'][t+1]-self.overlaps['variance'][t])
        diff += np.abs(self.overlaps['self_overlap'][t+1]-self.overlaps['self_overlap'][t])
        diff += np.abs(self.overlaps['teacher_student'][t+1]-self.overlaps['teacher_student'][t])
        diff += np.abs(self.overlaps['student_mean'][t+1]-self.overlaps['student_mean'][t])

        return diff

    def damp(self, new, old):
        '''
        Damping function.
        '''
        return (1-self.damping) * new + self.damping * old

    def iterate(self):
        '''
        Iterate the saddle-point equations.
        
        
        '''
        
        
        self._initialise()
        old_diff=diff=np.inf

        for t in range(self.max_steps):
            
            #print(self.hatoverlaps['student_mean'],self.hatoverlaps['variance'])
            
            self.hatoverlaps['variance'][t+1],self.hatoverlaps['self_overlap'][t+1],self.hatoverlaps['teacher_student'][t+1],self.hatoverlaps['student_mean'][t+1]=self.model._update_hatoverlaps(
                self.overlaps['variance'][t],self.overlaps['self_overlap'][t],
                                               self.overlaps['teacher_student'][t],self.overlaps['student_mean'][t])
            
            Vtmp, qtmp, mtmp ,htmp= self.model._update_overlaps(self.hatoverlaps['variance'][t+1],self.hatoverlaps['self_overlap'][t+1],self.hatoverlaps['teacher_student'][t+1],self.hatoverlaps['student_mean'][t+1])

            self.overlaps['variance'][t+1] = self.damp(Vtmp, self.overlaps['variance'][t])
            self.overlaps['self_overlap'][t+1] = self.damp(qtmp, self.overlaps['self_overlap'][t])
            self.overlaps['teacher_student'][t+1] = self.damp(mtmp, self.overlaps['teacher_student'][t])
            self.overlaps['student_mean'][t+1] = self.damp(htmp, self.overlaps['student_mean'][t])
            
            old_diff=diff
            diff = self._get_diff(t)
            
            if diff>old_diff:
                self.damping=(1+3*self.damping)/4

            if True:#self.verbose:
                print('t: {}, diff: {}, student mean: {}, teacher mean: {}, hhat{}, qhat{}'.format(t, diff,
                                                                                               self.overlaps['student_mean'][t+1],
                                                                                               self.model.data_model.h0,
                                                                                           self.hatoverlaps['student_mean'][t+1],
                                                                                                  self.hatoverlaps['self_overlap'][t+1]))
            if diff < self.tol:
            # If iterations converge, set status = 1
                if self.verbose:
                    print('Saddle point equations converged with t={} iterations'.format(t+1))

                self.status = 1
                break

        if t == self.max_steps-1:
            # If iterations didn't converge, set status = -1
            if self.verbose:
                print('Saddle point equations did not converge with t={} iterations. Keeping last values'.format(t+1))

            self.status = -1

        self.t_max = t+1

        self.overlaps['variance'] = self.overlaps['variance'][:t+1]
        self.overlaps['self_overlap'] = self.overlaps['self_overlap'][:t+1]
        self.overlaps['teacher_student'] = self.overlaps['teacher_student'][:t+1]
        self.overlaps['student_mean'] = self.overlaps['student_mean'][:t+1]
        
        self.hatoverlaps['variance'] = self.hatoverlaps['variance'][:t+1]
        self.hatoverlaps['self_overlap'] = self.hatoverlaps['self_overlap'][:t+1]
        self.hatoverlaps['teacher_student'] = self.hatoverlaps['teacher_student'][:t+1]
        self.hatoverlaps['student_mean'] = self.hatoverlaps['student_mean'][:t+1]

        self.test_error = self.model.get_test_error(self.overlaps['self_overlap'][-1],
                                                    self.overlaps['teacher_student'][-1],
                                                    self.overlaps['student_mean'][-1])

        self.train_loss = self.model.get_train_loss(self.overlaps['variance'][-1],
                                                    self.overlaps['self_overlap'][-1],
                                                    self.overlaps['teacher_student'][-1],
                                                    self.overlaps['student_mean'][-1])

    def get_info(self):
        info = {
            'hyperparameters': {
                'initialisation': self.init,
                'damping': self.damping,
                'max_steps': self.max_steps,
                'tolerance': self.tol
            }
        }
        if self.status != 0:
            info.update({
                'status': self.status,
                'convergence_time': self.t_max,
                'test_error': self.test_error,
                'train_loss': self.train_loss,
                'overlaps': {
                    'variance': self.overlaps['variance'][-1],
                    'self_overlap': self.overlaps['self_overlap'][-1],
                    'teacher_student': self.overlaps['teacher_student'][-1],
                    'student_mean': self.overlaps['student_mean'][-1]
                },
                'hatoverlaps': {
                    'variance': self.hatoverlaps['variance'][-1],
                    'self_overlap': self.hatoverlaps['self_overlap'][-1],
                    'teacher_student': self.hatoverlaps['teacher_student'][-1],
                    'student_mean': self.hatoverlaps['student_mean'][-1]
                }
            })
        return info