# -*- coding: utf-8 -*-
"""
@author: Amin Dhaou
"""

import numpy as np

def predict_traj(X0s, learner, horizon):
        """
        Predict a trajectory
        
        Parameters
        ----------
        X0s : array_like
            data
        learner :
            Model used to predict
        horizon : int
            Prediction horizon
        
        Returns
        -------
        traj : array_like
            trajectory
        """
        
        num_traj = X0s.shape[2]
        dim_x = X0s.shape[1]
        T =  X0s.shape[0]
        predict = lambda X,d : np.dot(np.linalg.matrix_power(learner.coef_,d),X)
        #initialize predictions 
        predictions = np.zeros((T,dim_x, num_traj)) + np.NaN
        for t in range(T):
            pred = []
            for i in range(t%horizon+1):
                pred.append(predict(X0s[t-i,:,:],i+1)) 
            predictions[t,:,:] = np.mean(np.array(pred),axis=0)
        traj = predictions[:,1:,:]
        return traj