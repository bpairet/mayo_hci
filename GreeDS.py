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

import torch

def GreeDS(algo,first_guess=None,spicy=False):
    """
        GreeDS(algo)
        Compute the GreeDS algorithm.
        Parameters
        ----------
        algo : instance of a subclass of mayonnaise_pipeline
        first_guess : str : location of a first guess for x, default=None 
        
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
    with torch.no_grad():
        if spicy:
            GreeDS_iteration = f_spicy_GreeDS
            if algo.FRIES:
                print('Not implemented warning: spicy-FRIES is not implemented yet, running regular spicy')
        else:
            GreeDS_iteration = f_GreeDS
        device = algo.data.device
        if first_guess:
            x_k = torch.from_numpy(vip.fits.open_fits(first_guess)).to(device)
        else:
            x_k = torch.zeros(algo.n,algo.n, device=device)
        iter_frames = torch.zeros(algo.parameters_algo['greedy_n_iter'],algo.n,algo.n, device=device)

        for r in range(algo.parameters_algo['greedy_n_iter']):
            ncomp = r + 1
            for l in range(algo.parameters_algo['greedy_n_iter_in_rank']):
                x_k1, xl = GreeDS_iteration(x_k,algo,ncomp)
                x_k = x_k1.clone()
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
    with torch.no_grad():
        device = algo.data.device
        X = x.expand(algo.t,algo.n,algo.n)
        

        R = algo.data - mayo_hci.cube_rotate_kornia(x.expand(algo.t,algo.n,algo.n),algo.pa_rotate_matrix)
        if algo.FRIES:
            L = (R.view(algo.t,algo.n*algo.n) @ algo.V[:,:ncomp] @ algo.V[:,:ncomp].T).reshape(algo.t,algo.n,algo.n)
        else:
            U,Sigma,V = torch.pca_lowrank(R.view(algo.t,algo.n*algo.n),q=ncomp,niter=4,center=False)
            L = (U @ torch.diag(Sigma) @ V.T).reshape(algo.t,algo.n,algo.n)
        
        #U_np, S_np, V_np = np.linalg.svd(R.reshape(algo.t,algo.n*algo.n).to('cpu').detach().numpy(),full_matrices=False)
        #U = torch.from_numpy(U_np).to(device)
        #Sigma = torch.from_numpy(S_np).to(device)
        #V = torch.from_numpy(V_np.T).to(device)
        #L = (U[:,:ncomp] @ torch.diag(Sigma[:ncomp]) @ V[:,:ncomp].T).reshape(algo.t,algo.n,algo.n)
        S = algo.data - L
        S_der = mayo_hci.cube_rotate_kornia(S,algo.pa_derotate_matrix)
        frame = torch.mean(S_der,axis=0)*algo.mask
        frame *= frame>0
        return frame, L

def f_spicy_GreeDS(x,algo,ncomp):
    """
        f_spicy_GreeDS(x,algo,ncomp)
        Compute a single iteration of the spicy-GreeDS algorithm.

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
            Current image produced by spicy-GreeDS.
        L : numpy.array
            Current speckle field estimated by spicy-GreeDS.
    """
    
    with torch.no_grad():
        device = algo.data.device
        X = x.expand(algo.t,algo.n,algo.n)

        R = algo.data - mayo_hci.cube_rotate_kornia(x.expand(algo.t,algo.n,algo.n),algo.pa_rotate_matrix).expand(2,algo.t,algo.n,algo.n)

        U,Sigma,V = torch.pca_lowrank(  0.5*(R[0]+mayo_hci.scale_cube(R[1] ,algo.contraction_matrix)).reshape(algo.t,algo.n*algo.n)  ,q=ncomp,niter=4,center=False)
        L = (U @ torch.diag(Sigma) @ V.T).reshape(algo.t,algo.n,algo.n)

        #U_np, S_np, V_np = np.linalg.svd(0.5*(R[0]+mayo_hci.scale_cube(R[1] ,algo.contraction_matrix)).reshape(algo.t,algo.n*algo.n).to('cpu').detach().numpy(),full_matrices=False)
        #U = torch.from_numpy(U_np).to(device)
        #Sigma = torch.from_numpy(S_np).to(device)
        #V = torch.from_numpy(V_np.T).to(device)
        #L = (U[:,:ncomp] @ torch.diag(Sigma[:ncomp]) @ V[:,:ncomp].T).reshape(algo.t,algo.n,algo.n)
        S_0 = algo.data[0] - L
        S_1 = algo.data[1] - mayo_hci.scale_cube(L ,algo.dilatation_matrix)
        S_der = 0.5*(mayo_hci.cube_rotate_kornia(S_0,algo.pa_derotate_matrix) + mayo_hci.cube_rotate_kornia(S_1,algo.pa_derotate_matrix))
        frame = torch.mean(S_der,axis=0)*algo.mask
        frame *= frame>0
        return frame, L

