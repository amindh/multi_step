#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 25 08:43:51 2022

@author: Amin Dhaou
"""

import numpy as np
    
def tensor_to_dataset(traj_tensor):
    """
    Transform a 3D array into 2D

    Parameters
    ----------
    traj_tensor : 3D arraay

    Returns
    -------
    mat : 2D array
    """
    mat = np.vstack(traj_tensor.transpose((0,2,1)))
    return mat


def rms_error(trajs_a, trajs_b):
    """
    Computes the NRMSE error

    Parameters
    ----------
    trajs_a : 
       trajectory which is a matrix [timesteps, dim_data]
                   or tensor [timesteps, dim_data, num_traj].
    trajs_b : 
        same as traj1.

    Returns
    -------
        NRMSE error.

    """
    def _rms_traj(traj1, traj2):
        """Helper function to compute the NRMSE error between two trajectories. """
        err = (traj1- traj2)
        sq_err = err*err
        rms_err =  np.sqrt(np.mean(sq_err))
        norm= np.sqrt(np.mean(traj2*traj2))
        if norm == 0:
            norm = 1
        return rms_err/norm
    # If more than one trajectory
    if len(trajs_a.shape) == 3: 
        rms = np.array([_rms_traj(trajs_a[:,:,n], trajs_b[:,:,n]) for n in range(trajs_a.shape[2])])
    else:
        rms = _rms_traj(trajs_a, trajs_b) 
    return rms