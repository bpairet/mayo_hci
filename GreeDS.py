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

from hciplot import plot_frames as plots

import mayo_hci


def GreeDS(algo,aggressive_GreeDS=False):
    print('--------------------------------------------------------')
    print('Starting GreeDS with ')
    print('      n_iter = ' + str(algo.parameters_algo['greedy_n_iter']))
    print('      mask_radius = ' + str(algo.parameters_algo['greedy_mask']))
    print('      n_iter_in_rank = ' + str(algo.parameters_algo['greedy_n_iter_in_rank']))
    print('--------------------------------------------------------')
    t,n,_ = algo.data.shape
    x_k = np.zeros((n,n))
    iter_frames = np.zeros([algo.parameters_algo['greedy_n_iter'],n,n])
    #eta = 1
    for r in range(algo.parameters_algo['greedy_n_iter']):
        ncomp = r + 1
        for l in range(algo.parameters_algo['greedy_n_iter_in_rank']):
            #x_k1 = (1-eta)*x_k + eta*f_GreeDS(x_k,algo.data,algo.angles,ncomp)
            x_k1, xl = f_GreeDS(x_k,algo,ncomp)
            x_k = np.copy(x_k1)
        iter_frames[r,:,:] = x_k1
    print('done, returning iter_frames')
    return iter_frames, xl


            #iter_frames, _,xl = algo_disk_1_rankIter(data_,self.angles,greedy_n_iter,r_mask_greedy,fraction_coeff,n_iter_in_rank,interpolation,low_frequency=False,plot_true=False,verbose=True)

def f_GreeDS(x,algo,ncomp):
    t,n,_ = algo.data.shape
    X = np.zeros((t,n,n))
    X[:,:,:] = x
    R = algo.data - mayo_hci.cube_rotate_kornia(X,algo.angles,algo.center_image)
    #R = algo.data - vip.preproc.cube_derotate(X,-algo.angles,interpolation='nearneig')
    U, S, V = np.linalg.svd(R.reshape(t,n*n), full_matrices=False)
    L = np.dot(U[:,:ncomp],np.dot(np.diag(S[:ncomp]),V[:ncomp,:])).reshape(t,n,n)
    S = algo.data - L
    #S_der = vip.preproc.cube_derotate(S,algo.angles,interpolation='nearneig')
    S_der = mayo_hci.cube_rotate_kornia(S,-algo.angles,algo.center_image)
    #frame = vip.var.mask_circle(np.mean(S_der,axis=0)*algo.mask,algo.parameters_algo['greedy_mask'])
    frame = np.mean(S_der,axis=0)*algo.mask
    frame *= frame>0
    return frame, L