# -*- coding: utf-8 -*-
"""
@author: Amin Dhaou
"""


import numpy as np

def predict_traj(X0s, learner, horizon):
        """
        Predict trajectories
        
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
        predictions : array_like
            trajectories
        """

        T =  X0s.shape[0]
        predict = lambda X,d : np.dot(np.linalg.matrix_power(learner.coef_,d),X)
        #initialize predictions 
        predictions = []
        for t in range(T-2):
            pred = []
            for i in range(t%horizon+1):
                pred.append(predict(X0s[t-i,:,:],i+1))
            predictions.append(np.array(pred)[:,1:,:])
        return predictions 