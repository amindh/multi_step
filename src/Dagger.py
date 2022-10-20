#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Amin Dhaou
"""
import numpy as np
from copy import deepcopy
from tqdm import tqdm
from sklearn.linear_model import LinearRegression
from src.utils.utils import tensor_to_dataset,rms_error

#%% DAD Algorithm multidim

class Dagger(object):
    "Class for the DAgger algorithm"

    def __init__(self, incr = False,res = False, horizon = 3):
        self.horizon = horizon
        self.incr = incr
        self.res = res

    def learn(self, X, predict_traj, learner, dagger_iters = 20):
        """
        Learn multi-step model

        Parameters
        ----------
        X : 
            Dataset.
        predict_traj :
            Method.
        learner : 
            Model (e.g Linear Regression).
        dagger_iters : int
            DAgger iterations. The default is 20.

        Returns
        -------
        train_errors : list
            Training errors at each iteration.
        C_history : list
            Matrices learned at each iteration.

        """
        # store reference to learner 
        if self.incr:
            self.step = 5
        else:
            self.step = self.horizon
        self.learner = learner
        self.T, self.dim, self.lag, self.num_traj = X.shape

        Xt1 = tensor_to_dataset(X[1:,:,0,:])
        Xt = np.concatenate([tensor_to_dataset(X[:-1,k,:,:]) for k in
                             range(self.dim)],axis=1)
        
        #add intercept
        Xt = np.insert(Xt, 0, 1, axis=1)
        Xt1_gt = Xt1.copy() # these will be ground truth targets
        
        X0s = np.concatenate([X[:,k,:,:] for k in range(self.dim)],axis=1)
        X0s = np.insert(X0s, 0, 1, axis=1) 

        # Initializing the Matrix 
        C_curr = np.zeros((self.dim*self.lag+1, self.dim*self.lag+1))
        C_curr[0,0] = 1
        for i in range(1,self.dim*self.lag+1):
            C_curr[i,i-1] = 1

        # Learning the coefficients
        self.learner.fit(Xt, Xt1)
        for k in range(self.dim):
            C_curr[1+k*self.lag,:] = deepcopy(self.learner.coef_[k,:])
            C_curr[1+k*self.lag,0] = deepcopy(self.learner.intercept_[k])
    

        self.learner.coef_ = deepcopy(C_curr)
        
        # Find the initial training error.
        errors = self.loss(X0s, self.learner.coef_)

        self.mean_train_pred_err = np.mean(errors)
        self.min_train_error = self.mean_train_pred_err
        self.min_train_error_model = deepcopy(self.learner.coef_)

        

        # Start the DaD Main loop
        train_errors = []
        C_history = [self.learner.coef_]
        
        curr_err = np.inf
        for i in tqdm(range(1, dagger_iters+1)): 
            Xpred = predict_traj(X0s,self.learner, self.step)
        
            L = self.loss(X0s,self.learner.coef_)
            self.mean_train_pred_err = np.mean(L)
            if self.mean_train_pred_err < self.min_train_error:
                self.min_train_error = self.mean_train_pred_err 
                self.min_train_error_model = deepcopy(self.learner.coef_) 
            
            err = self.mean_train_pred_err
            train_errors.append(err)
            
            #Add increment
            if self.incr:
                if curr_err-err > 0.00001  :
                    improvement = 1
                    curr_err = err
                    count = 0
                    X_pred_curr = deepcopy(Xpred)
                    Xt_curr = deepcopy(Xt)
                    Xt1_curr = deepcopy(Xt1)
                    learner_curr = deepcopy(self.learner.coef_)
                elif curr_err-err < 0.00001 and self.step < self.horizon and count == 10:
                    self.step = min(self.step+5, self.horizon)
                    count = 1
                    if improvement == 1:
                        Xpred = X_pred_curr
                        Xt= Xt_curr 
                        Xt1 = Xt1_curr
                        self.learner.coef_ = learner_curr
                elif curr_err-err < 0.00001 and count < 10:
                    count += 1
                elif count == 10 and self.step == self.horizon:
                    break
                
            # If Method_R, we have more than one trajectory
            if self.res:
                xt_hat = tensor_to_dataset(np.concatenate(Xpred,axis=0))
                xt_hat = np.insert(xt_hat, 0, 1, axis=1)
                ids = np.concatenate(
                    [[i+1]*Xpred[i].shape[0]  for i in range(len(Xpred)) ])
                
                Xt1_gt = tensor_to_dataset(np.array([X[k] for k in ids])[:,:,0,:])
            else:
                xt_hat = tensor_to_dataset(Xpred[:-2,:,:]) 
                xt_hat = np.insert(xt_hat, 0, 1, axis=1)
                Xt1_gt = tensor_to_dataset(X[2:,:,0,:])

            Xt = np.concatenate((Xt, xt_hat))
            Xt1 = np.concatenate((Xt1, Xt1_gt))
            train_xt = Xt; train_xt1 = Xt1;

            print("Fitting the model on the updated Dataset...")
            
            self.learner.fit(train_xt,train_xt1)
            for k in range(self.dim):
                C_curr[1+k*self.lag,:] = deepcopy(self.learner.coef_[k,:])
                C_curr[1+k*self.lag,0] = deepcopy(self.learner.intercept_[k])
        
            self.learner.coef_ = deepcopy(C_curr)
            C_history.append(self.learner.coef_)

        return train_errors, C_history

    def loss(self, X, C):
        """
        Compute the loss over a trajectory
        
        Parameters
        ----------
        X : array
            Dataset
        C : array
            Matrix.

        Returns
        -------
        test_pred_err : list
            list of error for each time step.

        """
        test_pred_err=[]
        for k in range(self.dim):
            for i in range(self.horizon-1):
                Cs = np.linalg.matrix_power(C, i+1)
                test_pred_err.append(rms_error(np.matmul(Cs,X[:-(i+1),:,:])[:,1+k*self.lag,:], 
                                   X[i+1:,1+k*self.lag,:]))    
        
        return test_pred_err
