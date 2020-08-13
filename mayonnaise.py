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
    assert ('conv' in parameters_algo), "KeyError: no conv, , no output produced..."
    try:
        parameters_algo['mask_center']
    except KeyError:
        parameters_algo['mask_center'] = 0
    assert ('scales' in parameters_algo), "KeyError: no scales, required for shearlets regularization, no output produced..."
    #if parameters_algo['min_objective'] == 'huber_loss':
    #    assert ('huber_param' in parameters_algo), "KeyError: no huber_param, required for huber_loss, no output produced..."
    try:
        parameters_algo['stochastic_gradient']
    except KeyError:
        parameters_algo['stochastic_gradient'] = False
    return parameters_algo


class mayonnaise_pipeline(object):
    def __init__(self,working_dir):
        self.working_dir = working_dir
        try:
            with open(self.working_dir+'parameters_algo.json', 'r') as read_file_parameters_algo:
                parameters_algo = json.load(read_file_parameters_algo)
        except FileNotFoundError:
            print('working_dir not found')
        self.data_name = parameters_algo['data_name']
        self.parameters_algo = verify_parameters_algo(parameters_algo)
        if 'data_path' in parameters_algo: 
            self.data,self.angles,self.psf = automatic_load_data(self.data_name,channel=self.parameters_algo['channel'],quick_look=0,crop=self.parameters_algo['crop'],dir=parameters_algo['data_path'])
        else:
            self.data,self.angles,self.psf = automatic_load_data(self.data_name,channel=self.parameters_algo['channel'],quick_look=0,crop=self.parameters_algo['crop'])
        try:
            with open(self.working_dir+'add_synthetic_signal.json', 'r') as read_file_add_synthetic_signal:
                add_synthetic_signal = json.load(read_file_add_synthetic_signal)
                self.data,self.synthetic_disc_planet = create_synthetic_data_with_disk_planet(self.data,self.angles,self.psf,add_synthetic_signal)
                print('Synthetic signal added to data')
        except FileNotFoundError:
            pass
        #self.data_name += '_'+str(self.parameters_algo['channel'])
        # the center corresponds to the crop in automatic_load_data
        self.t,self.n,_ = self.data.shape
        if 'center_image' in parameters_algo:
            self.center_image = tuple(parameters_algo['center_image'])
        else:
            self.center_image = False
        self.center_image, self.mask = get_rotation_center_and_mask(self.n,self.parameters_algo['mask_center'],self.center_image)
        self.kernel = np.fft.fft2(np.fft.fftshift(self.psf))
        self.matrix = self.data.reshape(self.t,self.n*self.n)
        self.run_GreeDS()
        self.xd = np.copy(self.GreeDS_frame)
        self.residuals = np.copy(self.GreeDS_frame)
        self.xp = np.zeros((self.n,self.n))
        self.define_optimization_function()
    def check_M_positive_semidefinite(self,n_tests):
        xMx = self.compute_xMx(self.S)
        assert(xMx >= 0), " xMx = "+str(xMx)+", M is not positive semidefinite! Values of delta or gamma are not suitable"
        for i in range(n_tests):
            x = [np.random.randn(*self.S[ii].shape) for ii in range(self.n_variables)]
            xMx = self.compute_xMx(x)
            assert(xMx >= 0), " xMx = "+str(xMx)+", M is not positive semidefinite! Values of delta or gamma are not suitable"
        print('For all the '+str(n_tests)+' xMx is positive, thus M seems positive semidefinite.')
    def compute_xMx(self,x):
        assert(len(x) == self.n_variables), "In compute_M : x does not have the right dimension"
        xMx = 0
        for ii in range(self.n_variables):
            xMx += np.sum(x[ii]* x[ii] - self.gamma*self.delta * x[ii]*self.L[ii](self.L_T[ii](x[ii])))
        return xMx
    def run_GreeDS(self,force_GreeDS=False):
        is_run_GreeDS = False
        greedy_n_iter = self.parameters_algo['greedy_n_iter']
        n_iter_in_rank = self.parameters_algo['greedy_n_iter_in_rank']
        r_mask_greedy =  self.parameters_algo['greedy_mask']
        if "aggressive_GreeDS" in self.parameters_algo:
            aggressive_GreeDS = self.parameters_algo['aggressive_GreeDS']
        else:
            aggressive_GreeDS = False
        saving_string = 'GreeDS_'+str(greedy_n_iter)+'_'+str(n_iter_in_rank)+'_'+str(r_mask_greedy)
        if not force_GreeDS:
            try:
                iter_frames = vip.fits.open_fits(self.working_dir+saving_string+'_iter_frames.fits')
                xl = vip.fits.open_fits(self.working_dir+saving_string+'_xl.fits')
            except FileNotFoundError:
                is_run_GreeDS = True
        else:
            is_run_GreeDS = True
        if is_run_GreeDS:
            '''
            fraction_coeff = 0.1
            interpolation = 'nearneig'
            if self.n%2 == 1:
                data_ = np.zeros((t,n+1,n+1))
                data_[:,:-1,:-1] = np.copy(self.data)
            else:
                data_  = np.copy(self.data)
            '''
            iter_frames, xl = mayo_hci.GreeDS(self,aggressive_GreeDS=aggressive_GreeDS)
            if not force_GreeDS: # force_GreeDS is used for bootstrap, we do not want to save the result
                vip.fits.write_fits(self.working_dir+saving_string+'_iter_frames.fits',iter_frames)
                vip.fits.write_fits(self.working_dir+saving_string+'_xl.fits',xl)
        self.GreeDS_frame = iter_frames[-1,:,:]
        self.xl = xl
    def set_disk_planet_regularization(self):
        self.shearletSystem = pyshearlab.SLgetShearletSystem2D(0,self.n,self.n, self.parameters_algo['scales'])
        self.Phi = lambda x: pyshearlab.SLsheardec2D(x, shearletSystem=self.shearletSystem)
        self.Phi_T = lambda x: pyshearlab.SLshearadjoint2D(x, shearletSystem=self.shearletSystem)
        if self.parameters_algo['conv']:
            self.conv_op = lambda x : A(x,self.kernel)
            self.adj_conv_op = lambda x : A_(x,self.kernel)
        else:
            self.conv_op = lambda x : x
            self.adj_conv_op = lambda x : x
        if self.parameters_algo['min_objective'] == 'l2_min':
            self.compute_loss = compute_l2_loss
        elif self.parameters_algo['min_objective'] == 'l1_min': # This is an approximation
            self.huber_delta = 0.001
            self.sigma_by_annulus = np.ones((self.n,self.n))
            self.compute_loss = lambda x : compute_normalized_huber_loss(x,self.huber_delta,torch.from_numpy(self.sigma_by_annulus))
        elif self.parameters_algo['min_objective'] == 'huber_loss':
            self.huber_parameters, self.sigma_by_annulus = get_huber_parameters(self)
            self.huber_delta,_ = self.huber_parameters
            self.compute_loss = lambda x : compute_normalized_huber_loss(x,self.huber_delta,torch.from_numpy(self.sigma_by_annulus))
        #elif self.parameters_algo['min_objective'] == 'generalized_huber_loss':
        #    assert ('huber_c1' in self.parameters_algo), "KeyError: huber_c1 not specified, no output produced..."
        #    assert ('huber_c2' in self.parameters_algo), "KeyError: huber_c2 not specified, no output produced..."
        #    self.compute_loss = lambda x : compute_generalized_huber_loss(x,self.parameters_algo['huber_param'],
        #                                                            self.parameters_algo['huber_c1'],self.parameters_algo['huber_c2'])
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
    def mayonnaise_pipeline_initialisation(self,Lip):
        if self.parameters_algo['min_objective'] == 'huber_loss':
            #delta, a = self.huber_parameters
            Lip *= np.max(1/self.sigma_by_annulus)
            #Lip *= np.max(self.sigma_by_annulus)
        #self.gamma = 1./Lip
        self.gamma = 1.3/Lip
        self.delta = 0.9/self.gamma
        #self.delta = 1./8/self.gamma
        self.norm_data = np.sqrt(np.sum(self.data**2))
        self.X = [0,0]
        self.S = [0, 0]
        self.Z = [0,0]
        self.regularization_disk = self.parameters_algo['regularization_disk']
        self.regularization_planet = self.parameters_algo['regularization_planet']
        if self.parameters_algo['regularization'] == 'lasso':
            self.regularization_disk *= self.delta
            self.regularization_planet *= self.delta
        self.convergence = np.zeros([self.parameters_algo['max_iter']])
        self.convergence_X = np.zeros([self.parameters_algo['max_iter']])
        self.convergence_Z = np.zeros([self.parameters_algo['max_iter']])
        self.n_iter = 0
        self.parameters_algo['stop-optim'] = False
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
        set_disk_regularization_parameter(self,parameter_disk)
        set_planet_regularization_parameter(self,parameter_planet)
    def mayonnaise_pipeline_iteration(self):
        previous_X = np.copy(self.X)
        previous_Z = np.copy(self.Z)
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
            self.convergence_X[self.n_iter] += np.sum( (previous_X[ii] - self.X[ii])**2 )
            self.convergence_Z[self.n_iter] += np.sum( (previous_Z[ii] - self.Z[ii])**2 )
        self.convergence[self.n_iter] = np.sqrt(self.convergence_X[self.n_iter] + self.convergence_Z[self.n_iter] )/self.norm_data/self.gamma
        del previous_X, previous_Z
        self.n_iter += 1
        print('\r at iteration '+str(self.n_iter)+', convergence is {:.5e}'.format(self.convergence[self.n_iter-1]), end='')
        if self.convergence[self.n_iter-1] < self.parameters_algo['tol']:
            self.parameters_algo['stop-optim'] = 'VAR_CONV'
        if self.n_iter >= self.parameters_algo['max_iter']:
            self.parameters_algo['stop-optim'] = 'MAX_ITER'
    def get_current_l1_norms(self):
        l1_norm_Phi_disk = np.sum(np.abs( self.Phi(self.X[0]) ))
        l1_norm_planet = np.sum(np.abs(self.X[1]))
        return l1_norm_Phi_disk, l1_norm_planet
    def get_rotation_and_mask_info(self):
        #center_coord, rotation_center_and_mask = get_rotation_center_and_mask(self.n,self.parameters_algo['mask_center'])
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
        data_overlay_rotation_center = self.data[0,:,:]*(rotation_center+0.5)/1.5
        return rotation_center_and_mask, data_overlay_rotation_center
    def solve_optim(self):
        while not self.parameters_algo['stop-optim']:
            self.mayonnaise_pipeline_iteration()
        print('Done with optimization')

class mca_disk_planet_mayonnaise_pipeline(mayonnaise_pipeline):
    def __init__(self,working_dir):
        super(mca_disk_planet_mayonnaise_pipeline, self).__init__(working_dir)
        #self.GreeDS_frame = np.zeros((self.n,self.n))
        self.set_disk_planet_regularization()
        self.mayonnaise_pipeline_initialisation()
        self.define_optimization_function()
        self.frame_data = np.copy(self.GreeDS_frame)
    def mayonnaise_pipeline_initialisation(self):
        Lip = 1.
        super(mca_disk_planet_mayonnaise_pipeline, self).mayonnaise_pipeline_initialisation(Lip)
        self.norm_data = np.sqrt(np.sum(self.GreeDS_frame**2))
        self.X = [self.xd,self.xp]
        self.S = [self.L[0](self.xd),self.L[1](self.xp)]
        self.Z = [self.xd,self.xp]
    def define_optimization_function(self):
        super(mca_disk_planet_mayonnaise_pipeline, self).define_optimization_function()
        self.n_variables = 2
        self.compute_grad = lambda : grad_MCA_pytorch(self.X[0],self.X[1],noisy_disk_planet=self.frame_data, 
                                                compute_loss=self.compute_loss,
                                                conv_op = self.conv_op, adj_conv_op=self.adj_conv_op,
                                                mask=self.mask)



class all_ADI_sequence_mayonnaise_pipeline(mayonnaise_pipeline):
    def __init__(self,working_dir):
        super(all_ADI_sequence_mayonnaise_pipeline, self).__init__(working_dir)
        self.set_disk_planet_regularization()
        self.mayonnaise_pipeline_initialisation()
        self.define_optimization_function()
    def mayonnaise_pipeline_initialisation(self):
        Lip = self.t
        if self.parameters_algo['stochastic_gradient']:
            Lip *= self.parameters_algo['stochastic_gradient']
        super(all_ADI_sequence_mayonnaise_pipeline, self).mayonnaise_pipeline_initialisation(Lip)
        self.U_L0,_,_ = randomized_svd(self.xl.reshape(self.t,self.n*self.n), n_components=self.parameters_algo['rank'], n_iter=5,transpose='auto')
        Low_rank_xl = (self.U_L0 @ self.U_L0.T @ self.xl.reshape(self.t,self.n*self.n) ).reshape(self.t,self.n,self.n)
        self.S_der = mayo_hci.cube_rotate_kornia(self.data - Low_rank_xl,-self.angles,self.center_image)
        self.xd = np.median(self.S_der,axis=0)*self.mask
        self.xd *= self.xd>0
        self.xp = np.zeros((self.n,self.n))
        self.X = [self.xd, self.xp, Low_rank_xl.reshape(self.t,self.n*self.n)]
        self.S = [self.L[0](self.xd),self.L[1](self.xp),self.L[2](Low_rank_xl.reshape(self.t,self.n*self.n))]
        self.Z = [self.xd,self.xp, Low_rank_xl.reshape(self.t,self.n*self.n)]
        self.norm_data = np.sqrt(np.sum(self.data**2))
    def define_optimization_function(self):
        super(all_ADI_sequence_mayonnaise_pipeline, self).define_optimization_function()
        self.n_variables = 3
        if self.parameters_algo['stochastic_gradient']:
            self.compute_grad = lambda : compute_MAYO_grad_stochastic(self.X[0],self.X[1],self.X[2],matrix=self.matrix,angles=self.angles,
                                                                    compute_loss=self.compute_loss,
                                                                    kernel=self.kernel,
                                                                    mask=self.mask,
                                                                    center_image=self.center_image,
                                                                    proportion_frames=self.parameters_algo['stochastic_gradient'])
        else:
            self.compute_grad = lambda : compute_cube_frame_conv_grad_pytorch(self.X[0],self.X[1],self.X[2],matrix=self.matrix,angles=self.angles,
                                                        compute_loss=self.compute_loss,
                                                        kernel=self.kernel,
                                                        mask=self.mask,
                                                        center_image=self.center_image)
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
    def internal_MCA_mayonnaise_pipeline_iteration(self,n_iter_mca):
        self.noisy_disk_planet = np.median(vip.preproc.cube_derotate(self.data - self.X[2].reshape(self.t,self.n,self.n),self.angles),axis=0)
        gamma_MCA = 1.
        if self.parameters_algo['min_objective'] == 'huber_loss':
            gamma_MCA /= np.max(1/self.sigma_by_annulus)
        #delta_MCA = 1./8/gamma_MCA
        delta_MCA = 0.9/gamma_MCA
        for k in range(n_iter_mca):
            previous_X = np.copy(self.X)
            previous_Z = np.copy(self.Z)
            print('\r at iteration '+str(self.n_iter)+', convergence is {:.5e}'.format(self.convergence[self.n_iter-1]) +', performing ' +str(k+1) + '/'+str(n_iter_mca)+' iterations of MCA.', end='')
            for ii in range(self.n_variables-1):
                self.X[ii] = self.prox_gamma_g[ii](self.Z[ii])
            temp_grad = self.compute_MCA_grad()
            grad = temp_grad[:-1]
            self.current_smooth_loss = temp_grad[-1]
            for ii in range(self.n_variables-1):
                v_temp = self.S[ii] - gamma_MCA*delta_MCA * self.L[ii](self.L_T[ii](self.S[ii])) + delta_MCA*self.L[ii](2*self.X[ii] - self.Z[ii] - gamma_MCA*grad[ii])
                self.S[ii] = self.prox_delta_h_star[ii](v_temp)
            for ii in range(self.n_variables-1):
                self.Z[ii] = self.X[ii] - gamma_MCA*grad[ii] - gamma_MCA*self.L_T[ii](self.S[ii])



class all_ADI_sequence_mayonnaise_pipeline_no_regul(mayonnaise_pipeline):
    def __init__(self,working_dir):
        super(all_ADI_sequence_mayonnaise_pipeline_no_regul, self).__init__(working_dir)
        self.set_disk_planet_regularization()
        self.mayonnaise_pipeline_initialisation()
        self.define_optimization_function()
        self.delta = 1
    def mayonnaise_pipeline_initialisation(self):
        Lip = self.t
        super(all_ADI_sequence_mayonnaise_pipeline_no_regul, self).mayonnaise_pipeline_initialisation(Lip)
        self.U_L0,_,_ = randomized_svd(self.xl.reshape(self.t,self.n*self.n), n_components=self.parameters_algo['rank'], n_iter=5,transpose='auto')
        Low_rank_xl = (self.U_L0 @ self.U_L0.T @ self.xl.reshape(self.t,self.n*self.n) ).reshape(self.t,self.n,self.n)
        self.X = [self.xd, Low_rank_xl.reshape(self.t,self.n*self.n)]
        self.S = [self.L[0](self.xd), self.L[1](Low_rank_xl.reshape(self.t,self.n*self.n))]
        self.Z = [self.xd, Low_rank_xl.reshape(self.t,self.n*self.n)]
        self.norm_data = np.sqrt(np.sum(self.data**2))
    def define_optimization_function(self):
        super(all_ADI_sequence_mayonnaise_pipeline_no_regul, self).define_optimization_function()
        self.n_variables = 2
        self.compute_grad = lambda : compute_cube_frame_grad_pytorch_no_regul(self.X[0],self.X[1],matrix=self.matrix,angles=self.angles,
                                                                compute_loss=self.compute_loss,
                                                                mask=self.mask,
                                                                center_image=self.center_image)
        self.proj_L_constraint = lambda x :self.U_L0 @ self.U_L0.T @ x
        self.noisy_disk_planet = self.GreeDS_frame
        self.L =  [lambda x : x, lambda x : x]
        self.L_T =  [lambda x : x, lambda x : x]
        self.prox_gamma_g = [positivity, positivity]
        self.prox_delta_h_star = [lambda x : x*0, 
                                    lambda x : x - self.delta*(self.proj_L_constraint(x/self.delta))]
        self.noisy_disk_planet = self.GreeDS_frame
