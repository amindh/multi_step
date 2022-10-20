#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Amin Dhaou
"""

#%%
import numpy as np
from sklearn.linear_model import LinearRegression

import deepcopy
from src.predict_methods import DaD, DaD_R, DaD_sub, DaD_agg
from src.Dagger import Dagger

#%%

#import data
# shape [T,num_var,num_lag,num_traj]
#T: length of TS; num_var:number of variables; num_lag: number of lags
#num_traj: number of trajectories
#X_train, X_val, X_test = ...

if __name__ == "__main__":
    
    #import your dataset
    #X_train, X_val, X_test = ...
    X_train, X_val, X_test = [np.random.randint(10,size=(50,2,1,30))]*3
    
    d_iter= 70

    H = 30

    dim = X_train.shape[1]

    if dim == 1:
        X0s_val= X_val[:,0,:,:]
    else: 
        X0s_val =  np.concatenate([X_val[:,k,:,:] for k in
                     range(dim)],axis=1)
    X0s_val = np.insert(X0s_val, 0, 1, axis=1)

    model = LinearRegression()
    
    #choose method 
    predict = DaD.predict_traj

    dad = Dagger(incr = False, horizon = H)
    train_errors, C_history = dad.learn( deepcopy(X_train), 
                    predict, model, dagger_iters = d_iter)
    print("best training loss")
    print(np.min(train_errors))