# -*- coding: utf-8 -*-
"""
@author: Amin Dhaou
"""

import numpy as np

def predict_traj(X0s, learner, horizon):
        """
        PRedict a trajectory
        
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
        power = 1
        for t in range(0, T):
            if power <= horizon:
                Xt = X0s[t+1-power,:,:]
                Xt1 = predict(Xt,power)
                power +=1
            else:
                power = 1
                Xt = X0s[t+1-power,:,:]
                Xt1 = predict(Xt,power)
                power+=1
            predictions[t,:,:] = Xt1
        traj = predictions[:,1:,:]  
        return traj