'''
mayonnaise.py 

Implementation of the MAYONNAISE pipeline from [PAI2020]


Notes
-----
[PAI2020] Pairet, Benoît, Faustine Cantalloube, and Laurent Jacques.
"MAYONNAISE: a morphological components analysis pipeline 
for circumstellar disks and exoplanets imaging in the near infrared." 
arXiv preprint arXiv:2008.05170 (2020).
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

import vip_hci as vip

import json

import numpy as np
from sklearn.decomposition import randomized_svd

from mayo_hci.automatic_load_data import automatic_load_data

from mayo_hci.operators import *
from mayo_hci.algo_utils import *
from mayo_hci.create_synthetic_data_with_disk_planet import *

import torch
from shearlet_dec_ajd_dec_pytorch import SLsheardec2D_pytorch, SLshearadjoint2D_pytorch

def verify_parameters_algo(parameters_algo):
    assert ('min_objective' in parameters_algo), "KeyError: no min_objective specified, no output produced..."
    assert ('rank' in parameters_algo), "KeyError: no rank specified, no output produced..."
    assert ('regularization_disk' in parameters_algo), "KeyError: no regularization_disk specified, no output produced..."
    assert ('regularization_planet' in parameters_algo), "KeyError: no raregularization_planettio_d_and_p specified, no output produced..."
    assert ('tol' in parameters_algo), "KeyError: no tol specified, no output produced..."
    assert ('max_iter' in parameters_algo), "KeyError: no max_iter specified, no output produced..."
    assert ('regularization' in parameters_algo), "KeyError: no regularization specified, no output produced..."
    assert ('greedy_n_iter' in parameters_algo), "KeyError: no greedy_n_iter, no output produced..."
    assert ('greedy_n_iter_in_rank' in parameters_algo), "KeyError: no greedy_n_iter_in_rank, , no output produced..."
    assert ('greedy_mask' in parameters_algo), "KeyError: no greedy_mask, , no output produced..."
    #assert ('conv' in parameters_algo), "KeyError: no conv, , no output produced..."
    try:
        parameters_algo['mask_center']
    except KeyError:
        parameters_algo['mask_center'] = 0
    try:
        parameters_algo['stochastic']
    except KeyError:
        parameters_algo['stochastic'] = False
    assert ('scales' in parameters_algo), "KeyError: no scales, required for shearlets regularization, no output produced..."
    return parameters_algo


class mayonnaise_pipeline(object):
    '''
    Initialize MAYO from the file parameters_algo.json in working_dir
    Performs operations 1 to 6 of the MAYO pipeline (Algorithm 2 in Pairet etal 2020)
    Differnt Child classes will solve either problem 27, D1 or D2 from Pairet etal 2020. 
    Parameters
    ----------
    working_dir : str
        working directory, containing the parameters_algo.json file and the add_synthetic_signal.json 
        when mayo runs on synthetic data
    '''
    def __init__(self,working_dir):
        #Check if we have a gpu, otherwise, set self.device to cpu:
        if torch.cuda.is_available():
            gpu_memory_map = get_gpu_memory_map()
            if gpu_memory_map[0] < gpu_memory_map[1]:
                self.device = torch.device('cuda:0')
            else:
                self.device = torch.device('cuda:1')
        else:
            self.device = torch.device('cpu')
        self.working_dir = working_dir
        try:
            with open(self.working_dir+'parameters_algo.json', 'r') as read_file_parameters_algo:
                parameters_algo = json.load(read_file_parameters_algo)
        except FileNotFoundError:
            print('working_dir not found')
        self.data_name = parameters_algo['data_name']
        self.parameters_algo = verify_parameters_algo(parameters_algo) 
        # if no data_path is given, use the default path
        if 'data_path' in parameters_algo: 
            self.data,self.angles,self.psf,self.center_image = automatic_load_data(self.data_name,self.device,channel=self.parameters_algo['channel'],crop=self.parameters_algo['crop'],data_dir=parameters_algo['data_path'])
        else:
            self.data,self.angles,self.psf,self.center_image = automatic_load_data(self.data_name,self.device,channel=self.parameters_algo['channel'],crop=self.parameters_algo['crop'])
        self.t,self.n,_ = self.data.shape
        self.kernel = torch.fft.fft2(torch.fft.fftshift(self.psf))
        self.data_np = self.data.cpu().detach().numpy()
        #self.psf = self.psf[self.n//2-10:self.n//2+10,self.n//2-10:self.n//2+10]
        #if 'center_image' in parameters_algo:
        #    self.center_image = tuple(parameters_algo['center_image'])
        #else:
        #    self.center_image = False
        mask_np = get_mask(self.n,self.parameters_algo['mask_center'],self.center_image)
        self.mask = torch.from_numpy(mask_np).to(self.device)
        dtype = torch.FloatTensor # todo : check how to deal with this
        # Define the rotation matrices:
        self.pa_rotate_matrix = get_all_rotation_matrices(self.angles,self.center_image,dtype).to(self.device)
        self.pa_derotate_matrix = get_all_rotation_matrices(-self.angles,self.center_image,dtype).to(self.device)
        #check if there is any synthetic data to add (only used to create synthetic data to test mayo)
        try:
            with open(self.working_dir+'add_synthetic_signal.json', 'r') as read_file_add_synthetic_signal:
                add_synthetic_signal = json.load(read_file_add_synthetic_signal)
                self.data_np,self.synthetic_disc_planet = create_synthetic_data_with_disk_planet(self.data_np,self.pa_rotate_matrix.to('cpu'),self.kernel.cpu().detach().numpy(),add_synthetic_signal)
                self.data = torch.from_numpy(self.data_np).to(self.device)
                print('Synthetic signal added to data')
        except FileNotFoundError:
            pass
        self.matrix = self.data.reshape(self.t,self.n*self.n)
        if 'ref_cube' in self.parameters_algo:
            try:
                if 'data_path' in parameters_algo:
                    path_ref_cube = parameters_algo['data_path']+self.data_name+'/'+self.parameters_algo['ref_cube'][self.parameters_algo['channel']]
                else:
                    print('warning: we need to fix the default data path')
                    path_ref_cube = "D:/HCImaging/Data/"+self.data_name+'/'+self.parameters_algo['ref_cube'][self.parameters_algo['channel']]
                
                ref_cube = mayo_hci.open_fits(path_ref_cube,device='cpu')
                n_frames_ref,_,_ = ref_cube.shape
                _,_,V_T = np.linalg.svd(ref_cube.view(n_frames_ref,self.n*self.n),full_matrices=False)
                self.V = torch.from_numpy(V_T.T).to(self.device)
            except FileNotFoundError:
                raise FileNotFoundError("Ref cube file not found!")
            self.FRIES = True
        else:
            self.FRIES = False
        self.run_GreeDS() # step 3 in algorithm 2 from Pairet etal 2020
        self.xd = self.GreeDS_frame.clone()
        self.residuals = self.GreeDS_frame.clone()
        self.xp = torch.zeros((self.n,self.n), device=self.device)
        if self.FRIES:
            self.V = self.V[:,:self.parameters_algo['rank']]
        else:
            U_L0_np,_,_ = randomized_svd(self.xl.reshape(self.t,self.n*self.n).cpu().detach().numpy(), n_components=self.parameters_algo['rank'], n_iter=5,transpose='auto')
            self.U_L0 = torch.from_numpy(U_L0_np).to(self.device)
        self.define_optimization_function() #dummy definition of function, redifined in child classes
    def check_LLT_smallerOne(self,n_tests): 
        '''
         Check that the norm of LL_T is smaller than 1/(delta gamma), see PD3O from Yan 2018 for details.
        '''
        max_norm_LLTx = 0
        norm_LLTx = 0
        #for ii in range(self.n_variables):
        #    x = self.S[ii]/np.sqrt(np.sum(self.S[ii]**2))
        #    norm_LLTx += np.sum( (self.L[ii](self.L_T[ii](x)) )**2 )
        #norm_LLTx = np.sqrt( norm_LLTx )
        #if max_norm_LLTx < norm_LLTx:
        #    max_norm_LLTx = norm_LLTx
        for i in range(n_tests):
            x = [torch.rand(*self.S[ii].shape, device=self.device) for ii in range(self.n_variables)]
            norm_LLTx = 0
            for ii in range(self.n_variables):
                norm_LLTx += ( (self.L[ii](self.L_T[ii](x[ii])) )**2 ).sum()
            norm_LLTx = torch.sqrt( norm_LLTx )
            if max_norm_LLTx < norm_LLTx:
                max_norm_LLTx = norm_LLTx
        if max_norm_LLTx > 1./(self.gamma*self.delta):
            print(" || LLTx ||(delta gamma) = "+str(norm_LLTx*(self.gamma*self.delta))+" ! Values of delta is changed")
            #self.delta = 0.5/(self.gamma*(max_norm_LLTx))
            self.delta = 0.75/(self.gamma*(max_norm_LLTx))
        else:
            print('For all the '+str(n_tests)+' LLTx was good ( || LLTx ||(delta gamma) = '+str(norm_LLTx*(self.gamma*self.delta))+'). delta seems allright, maybe too low.')
        print(max_norm_LLTx)
    def check_M_positive_semidefinite(self,n_tests): 
        '''
         The matrix xMx (as computed in compute_xMx) must be positive definite for PD3O from Yan 2018 to converge.
        '''
        xMx = self.compute_xMx(self.S)
        assert(xMx >= 0), " xMx = "+str(xMx)+", M is not positive semidefinite! Values of delta or gamma are not suitable"
        for i in range(n_tests):
            x = [torch.rand(*self.S[ii].shape, device=self.device) for ii in range(self.n_variables)]
            xMx = self.compute_xMx(x)
            assert(xMx >= 0), " xMx = "+str(xMx)+", M is not positive semidefinite! Values of delta or gamma are not suitable"
        print('For all the '+str(n_tests)+' xMx is positive, thus M seems positive semidefinite.')
    def compute_xMx(self,x):
        assert(len(x) == self.n_variables), "In compute_M : x does not have the right dimension"
        xMx = 0
        for ii in range(self.n_variables):
            xMx += (x[ii]* x[ii] - self.gamma*self.delta * x[ii]*self.L[ii](self.L_T[ii](x[ii]))).sum()
        return xMx
    def run_GreeDS(self,force_GreeDS=False,spicy=False):
        '''
        run_GreeDS(self,force_GreeDS=False)
        
        Wrapper around GreeDS,  runs if GreeDS has not run before or force_GreeDS=True, then saves the results.
        						loads results otherwise
        '''
        is_run_GreeDS = False
        greedy_n_iter = self.parameters_algo['greedy_n_iter']
        n_iter_in_rank = self.parameters_algo['greedy_n_iter_in_rank']
        r_mask_greedy =  self.parameters_algo['greedy_mask']
        if "GreeDS_first_guess" in self.parameters_algo:
            first_guess = self.parameters_algo['GreeDS_first_guess']
        else:
            first_guess = None
        saving_string = 'GreeDS_'+str(greedy_n_iter)+'_'+str(n_iter_in_rank)+'_'+str(r_mask_greedy)
        if not force_GreeDS:
            try:
                iter_frames = torch.from_numpy(vip.fits.open_fits(self.working_dir+saving_string+'_iter_frames.fits')).to(self.device)
                xl = torch.from_numpy(vip.fits.open_fits(self.working_dir+saving_string+'_xl.fits')).to(self.device)
            except FileNotFoundError:
                is_run_GreeDS = True
        else:
            is_run_GreeDS = True
        if is_run_GreeDS:
            iter_frames, xl = mayo_hci.GreeDS(self,first_guess=first_guess,spicy=spicy)
            if not force_GreeDS: # force_GreeDS is used for bootstrap, we do not want to save the result
                vip.fits.write_fits(self.working_dir+saving_string+'_iter_frames.fits',iter_frames.cpu().detach().numpy())
                vip.fits.write_fits(self.working_dir+saving_string+'_xl.fits',xl.cpu().detach().numpy())
        self.GreeDS_frame = iter_frames[-1,:,:]
        self.xl = xl
    
    def set_disk_planet_regularization(self):
        '''
         Defines the loss and regularization functions used in mayo from self.parameters_algo
         Pairet etal 2020 shows that using the Huber Loss is better than both l1 and l2 norms
         and should be used by default.
        '''
        self.shearletSystem = pyshearlab.SLgetShearletSystem2D(0,self.n,self.n, self.parameters_algo['scales'])
        self.shearlets_pytorch = torch.tensor(self.shearletSystem["shearlets"], device=self.device)
        self.Phi = lambda x: SLsheardec2D_pytorch(x, shearlets=self.shearlets_pytorch)
        self.Phi_T = lambda x: SLshearadjoint2D_pytorch(x, shearlets=self.shearlets_pytorch)
        if self.parameters_algo['conv']:
            self.conv_op = lambda x : A_pytorch(x,self.kernel)
            self.adj_conv_op = lambda x : A_adj_pytorch(x,self.kernel)
        else:
            self.conv_op = lambda x : x
            self.adj_conv_op = lambda x : x
        if self.parameters_algo['min_objective'] == 'l2_min':
            self.compute_loss = compute_l2_loss
        elif self.parameters_algo['min_objective'] == 'l1_min': # This is an approximation
            self.huber_delta = 0.001
            self.sigma_by_annulus = torch.ones((self.n,self.n), device=self.device)
            self.compute_loss = lambda x : compute_normalized_huber_loss(x,self.huber_delta,self.sigma_by_annulus)
        elif self.parameters_algo['min_objective'] == 'huber_loss':
            self.huber_parameters, sigma_by_annulus_numpy = get_huber_parameters(self)
            self.sigma_by_annulus = torch.from_numpy(sigma_by_annulus_numpy).to(self.device)
            self.huber_delta,_ = self.huber_parameters
            self.compute_loss = lambda x : compute_normalized_huber_loss(x,self.huber_delta,self.sigma_by_annulus)
        else:
            print('min_objective not recognized, no output produced...')
            raise Exception
    
    def define_optimization_function(self):
        self.n_variables = 2
        self.compute_grad = lambda: [0,0,0]
        if self.parameters_algo['regularization'] == 'lasso':
            self.prox_basis_disk = lambda x: soft_thresh(x, _lambda=self.regularization_disk)
            self.prox_basis_planet = lambda x : soft_thresh(x, param=self.regularization_planet)
        elif self.parameters_algo['regularization'] == 'constraint':
            self.prox_basis_disk = lambda x : frame_euclidean_proj_l1ball(x, _lambda=self.regularization_disk)
            self.prox_basis_planet = lambda x : frame_euclidean_proj_l1ball(x, _lambda=self.regularization_planet)
        self.L =  [lambda x : self.Phi(x), lambda x : x]
        self.L_T =  [lambda x : self.Phi_T(x), lambda x : x]
        self.prox_gamma_g = [positivity, positivity]
        self.prox_delta_h_star = [lambda x : x - self.delta*(self.prox_basis_disk(x/self.delta)), 
                                    lambda x : x - self.delta*(self.prox_basis_planet(x/self.delta))]
    
    def mayonnaise_pipeline_initialisation(self):
        if self.parameters_algo['min_objective'] == 'huber_loss':
            self.Lip *= torch.max(1/self.sigma_by_annulus)
        self.norm_data = torch.sqrt((self.data**2).sum())
        self.X = [0,0]
        self.S = [0, 0]
        self.Z = [0,0]
        self.regularization_disk = self.parameters_algo['regularization_disk']
        self.regularization_planet = self.parameters_algo['regularization_planet']
        if self.parameters_algo['regularization'] == 'lasso':
            self.regularization_disk *= self.delta
            self.regularization_planet *= self.delta
        self.convergence = torch.zeros([self.parameters_algo['max_iter']], device=self.device)
        self.convergence_X = torch.zeros([self.parameters_algo['max_iter']], device=self.device)
        self.convergence_Z = torch.zeros([self.parameters_algo['max_iter']], device=self.device)
        self.n_iter = 0
        self.parameters_algo['stop-optim'] = False
        # Parameters for the optimization:
        self.gamma = 1.1/self.Lip
        self.delta = 1./self.gamma*100
    
    def set_disk_regularization_parameter(self,parameter_disk):
        self.parameters_algo['regularization_disk'] = parameter_disk
        self.regularization_disk = self.parameters_algo['regularization_disk']
        if self.parameters_algo['regularization'] == 'lasso':
            self.regularization_disk *= self.gamma
    
    def set_planet_regularization_parameter(self,parameter_planet):
        self.parameters_algo['regularization_planet'] = parameter_planet
        self.regularization_planet = self.parameters_algo['regularization_planet']
        if self.parameters_algo['regularization'] == 'lasso':
            self.regularization_planet *= self.gamma
    
    def set_regularization_parameters(self,parameter_disk,parameter_planet):
        self.set_disk_regularization_parameter(parameter_disk)
        self.set_planet_regularization_parameter(parameter_planet)
    
    def get_rotation_and_mask_info(self):
        '''
         Returns the information about the coroagraphic mask and the center of rotation
         used by mayo. 
        ''' 
        center_coord = self.center_image
        rotation_center_and_mask = self.mask
        cx, cy = center_coord
        if (cx*1.0).is_integer():
            ind_x = int(cx)
        else:
            ind_x = np.array([int(cx), int(cx) + 1 ])
        if (cy*1.0).is_integer():
            ind_y = int(cy)
        else:
            ind_y = np.array([int(cy), int(cy) + 1 ])
        print(ind_x)
        print(ind_y)
        rotation_center = np.zeros((self.n,self.n))
        rotation_center[ind_x,ind_y] = 1.
        rotation_center_and_mask = rotation_center_and_mask*1. + rotation_center
        data_overlay_rotation_center = self.data_np[0,:,:]*(rotation_center+0.5)/1.5
        return rotation_center_and_mask, data_overlay_rotation_center
    
    def mayonnaise_pipeline_iteration(self):
        '''
        A single iteration of the Primal-Dual Three-Operator splitting (PD3O) algorithm
        presented in Yan 2018, and used to solve the unmixing optimization problem of MAYO
        If convergence or max iter is reached, self.parameters_algo['stop-optim'] is set to
        'VAR_CONV' or 'MAX_ITER'
        '''
        previous_X = [self.X[ii].detach().clone() for ii in range(self.n_variables)]
        previous_Z = [self.Z[ii].detach().clone() for ii in range(self.n_variables)]
        for ii in range(self.n_variables):
            self.X[ii] = self.prox_gamma_g[ii](self.Z[ii])
        temp_grad = self.compute_grad()
        grad = temp_grad[:-1]
        self.current_smooth_loss = temp_grad[-1]
        for ii in range(self.n_variables):
            v_temp = self.S[ii] - self.gamma*self.delta * self.L[ii](self.L_T[ii](self.S[ii])) + self.delta*self.L[ii](2*self.X[ii] - self.Z[ii] - self.gamma*grad[ii])
            self.S[ii] = self.prox_delta_h_star[ii](v_temp)
        for ii in range(self.n_variables):
            self.Z[ii] = self.X[ii] - self.gamma*grad[ii] - self.gamma*self.L_T[ii](self.S[ii])
        self.convergence_X[self.n_iter] = 0.
        self.convergence_Z[self.n_iter] = 0.
        for ii in range(self.n_variables):
            self.convergence_X[self.n_iter] += ( (previous_X[ii] - self.X[ii])**2 ).sum()
            self.convergence_Z[self.n_iter] += ( (previous_Z[ii] - self.Z[ii])**2 ).sum()
        self.convergence[self.n_iter] = torch.sqrt(self.convergence_X[self.n_iter] + self.convergence_Z[self.n_iter] )/self.norm_data/self.gamma
        del previous_X, previous_Z
        self.n_iter += 1
        if self.n_iter%120==0:
            print('\r at iteration '+str(self.n_iter)+', convergence is {:.5e}'.format(self.convergence[self.n_iter-1]), end='')
        if self.convergence[self.n_iter-1] < self.parameters_algo['tol']:
            self.parameters_algo['stop-optim'] = 'VAR_CONV'
        if self.n_iter >= self.parameters_algo['max_iter']:
            self.parameters_algo['stop-optim'] = 'MAX_ITER'
    
    def solve_optim(self):
        '''
        Solve optimization problem from Pairet etal. 2020, by calling mayonnaise_pipeline_iteration
        until self.parameters_algo['stop-optim'] is True 
        '''
        while not self.parameters_algo['stop-optim']:
            self.mayonnaise_pipeline_iteration()
        print('Done with optimization')


class all_ADI_sequence_mayonnaise_pipeline(mayonnaise_pipeline):
    '''
    Main instance of MAYO, solves optimization problem 27 from Pairet etal 2020
    '''
    def __init__(self,working_dir):
        super(all_ADI_sequence_mayonnaise_pipeline, self).__init__(working_dir)
        self.set_disk_planet_regularization()
        self.mayonnaise_pipeline_initialisation()
        self.define_optimization_function()
        # We check that optimization parameters are set so that the convergence conditions are met:
        self.check_LLT_smallerOne(10)
        self.check_M_positive_semidefinite(10)

    def mayonnaise_pipeline_initialisation(self):
        self.Lip = self.t
        super(all_ADI_sequence_mayonnaise_pipeline, self).mayonnaise_pipeline_initialisation()
        if self.FRIES:
            Low_rank_xl = (self.xl.view(self.t,self.n*self.n) @ self.V @ self.V.T).view(self.t,self.n,self.n)
        else:
            Low_rank_xl = (self.U_L0 @ self.U_L0.T @ self.xl.view(self.t,self.n*self.n) ).view(self.t,self.n,self.n)
        self.S_der = mayo_hci.cube_rotate_kornia(self.data - Low_rank_xl,self.pa_derotate_matrix)
        self.xd = torch.mean(self.S_der,axis=0)*self.mask
        self.xd *= self.xd>0
        self.xp = torch.zeros((self.n,self.n), device=self.device)
        self.X = [self.xd, self.xp, Low_rank_xl.reshape(self.t,self.n*self.n)]
        self.S = [self.L[0](self.xd),self.L[1](self.xp),self.L[2](Low_rank_xl.reshape(self.t,self.n*self.n))]
        self.Z = [self.xd,self.xp, Low_rank_xl.reshape(self.t,self.n*self.n)]
        self.norm_data = torch.sqrt((self.data**2).sum())
    def define_optimization_function(self):
        super(all_ADI_sequence_mayonnaise_pipeline, self).define_optimization_function()
        self.n_variables = 3
        self.rotated_data = mayo_hci.cube_rotate_kornia(self.data,self.pa_derotate_matrix)
        self.compute_grad = lambda : compute_rotatedSpeckles_conv_grad_pytorch(self.X[0],self.X[1],self.X[2],
                                               rotated_data=self.rotated_data,
                                               pa_derotate_matrix=self.pa_derotate_matrix,
                                               compute_loss=self.compute_loss,
                                               kernel=self.kernel,
                                                mask=self.mask)
        if self.FRIES:
            self.proj_L_constraint = lambda x : x @ self.V @ self.V.T
        else:
            self.proj_L_constraint = lambda x :self.U_L0 @ self.U_L0.T @ x
        self.noisy_disk_planet = self.GreeDS_frame
        self.L =  [lambda x : self.Phi(x), lambda x : x, lambda x : x]
        self.L_T =  [lambda x : self.Phi_T(x), lambda x : x, lambda x : x]
        self.prox_gamma_g = [positivity, positivity, positivity]
        self.prox_delta_h_star = [lambda x : x - self.delta*(self.prox_basis_disk(x/self.delta)), 
                                    lambda x : x - self.delta*(self.prox_basis_planet(x/self.delta)),
                                    lambda x : x - self.delta*(self.proj_L_constraint(x/self.delta))]
        self.noisy_disk_planet = self.GreeDS_frame
        self.compute_MCA_grad = lambda : grad_MCA_pytorch(self.X[0],self.X[1],noisy_disk_planet=self.noisy_disk_planet, 
                                                compute_loss=self.compute_loss,
                                                conv_op = self.conv_op, adj_conv_op=self.adj_conv_op,
                                                mask=self.mask)

class all_ADI_sequence_mayonnaise_pipeline_no_regul(mayonnaise_pipeline):
    '''
     Non-regularized version of MAYO, solves optimization problem D1 from Pairet etal 2020
    '''
    def __init__(self,working_dir):
        super(all_ADI_sequence_mayonnaise_pipeline_no_regul, self).__init__(working_dir)
        self.set_disk_planet_regularization()
        self.mayonnaise_pipeline_initialisation()
        self.define_optimization_function()
        self.delta = 1
        # We check that optimization parameters are set so that the convergence conditions are met:
        self.check_LLT_smallerOne(10)
        self.check_M_positive_semidefinite(10)

    def mayonnaise_pipeline_initialisation(self):
        self.Lip = self.t
        super(all_ADI_sequence_mayonnaise_pipeline_no_regul, self).mayonnaise_pipeline_initialisation()
        if self.FRIES:
            Low_rank_xl = (self.xl.view(self.t,self.n*self.n) @ self.V @ self.V.T).view(self.t,self.n,self.n)
        else:
            Low_rank_xl = (self.U_L0 @ self.U_L0.T @ self.xl.view(self.t,self.n*self.n) ).view(self.t,self.n,self.n)
        self.X = [self.xd, Low_rank_xl.reshape(self.t,self.n*self.n)]
        self.S = [self.L[0](self.xd), self.L[1](Low_rank_xl.reshape(self.t,self.n*self.n))]
        self.Z = [self.xd, Low_rank_xl.reshape(self.t,self.n*self.n)]
        self.norm_data = torch.sqrt((self.data**2).sum())
    def define_optimization_function(self):
        super(all_ADI_sequence_mayonnaise_pipeline_no_regul, self).define_optimization_function()
        self.n_variables = 2
        self.compute_grad = lambda : compute_cube_frame_grad_pytorch_no_regul(self.X[0],self.X[1],data=self.data,
                                                                pa_rotate_matrix=self.pa_rotate_matrix,
                                                                compute_loss=self.compute_loss,
                                                                mask=self.mask)
        if self.FRIES:
            self.proj_L_constraint = lambda x : x @ self.V @ self.V.T
        else:
            self.proj_L_constraint = lambda x :self.U_L0 @ self.U_L0.T @ x
        self.noisy_disk_planet = self.GreeDS_frame
        self.L =  [lambda x : x, lambda x : x]
        self.L_T =  [lambda x : x, lambda x : x]
        self.prox_gamma_g = [positivity, positivity]
        self.prox_delta_h_star = [lambda x : x*0, 
                                    lambda x : x - self.delta*(self.proj_L_constraint(x/self.delta))]
        self.noisy_disk_planet = self.GreeDS_frame
