'''
GreeDS algorithm from Pairet etal 2020

'''

'''
 MAYO pipeline, from Pairet et al. 2020
    Copyright (C) 2020, Benoit Pairet

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.
'''

import numpy as np
import vip_hci as vip
import mayo_hci


def GreeDS(algo):
    """
        GreeDS(algo)
        Compute the GreeDS algorithm.
        Parameters
        ----------
        algo : instance of a subclass of mayonnaise_pipeline
        
        Returns
        -------
        iter_frames : numpy.array
            Images produced by GreeDS for ranks from 0 to n_iter.
        xl : numpy.array
            Speckle field estimated by GreeDS.
    """
    print('--------------------------------------------------------')
    print('Starting GreeDS with ')
    print('      n_iter = ' + str(algo.parameters_algo['greedy_n_iter']))
    print('      mask_radius = ' + str(algo.parameters_algo['greedy_mask']))
    print('      n_iter_in_rank = ' + str(algo.parameters_algo['greedy_n_iter_in_rank']))
    print('--------------------------------------------------------')
    print('VIP rotation')
    t,n,_ = algo.data.shape
    x_k = np.zeros((n,n))
    iter_frames = np.zeros([algo.parameters_algo['greedy_n_iter'],n,n])

    for r in range(algo.parameters_algo['greedy_n_iter']):
        ncomp = r + 1
        for l in range(algo.parameters_algo['greedy_n_iter_in_rank']):
            x_k1, xl = f_GreeDS(x_k,algo,ncomp)
            x_k = np.copy(x_k1)
        iter_frames[r,:,:] = x_k1
    
    print('done, returning iter_frames')
    return iter_frames, xl



def f_GreeDS(x,algo,ncomp):
    """
        f_GreeDS(x,algo,ncomp)
        Compute a single iteration of the GreeDS algorithm.

        Parameters
        ----------
        x : numpy.array
            current estimate of the circumstellar signal
        algo : instance of a subclass of mayonnaise_pipeline

        ncomp: int
            rank of the speckle field component (xl)

        Returns
        -------
        frame : numpy.array
            Current image produced by GreeDS.
        L : numpy.array
            Current speckle field estimated by GreeDS.
    """

    t,n,_ = algo.data.shape
    X = np.zeros((t,n,n))
    X[:,:,:] = x

    R = algo.data - mayo_hci.cube_rotate_kornia(X,algo.rotation_matrices)
    U, S, V = np.linalg.svd(R.reshape(t,n*n), full_matrices=False)
    L = np.dot(U[:,:ncomp],np.dot(np.diag(S[:ncomp]),V[:ncomp,:])).reshape(t,n,n)
    S = algo.data - L
    S_der = mayo_hci.cube_rotate_kornia(S,algo.inv_rotation_matrices)
    frame = np.mean(S_der,axis=0)*algo.mask
    frame *= frame>0
    return frame, L