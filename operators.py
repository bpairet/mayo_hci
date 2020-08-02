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
import torch
import kornia

import pyshearlab
import pywt

from sklearn.decomposition import randomized_svd
from simplex_projection import euclidean_proj_l1ball

#from mayo_hci.algo_utils import get_donut

def prox_general_frame(x,_lambda,prox_operator,L,L_T,nu,tol,max_iter=100):
    if tol == 0:
        tol = 0.1
    n_iter = 0
    u = 0#np.copy(L(x))
    p = np.copy(x)
    converged = False
    while not converged:
        n_iter += 1
        temp = u + L(p)
        u_1 = ( temp -  prox_operator(temp,_lambda) )
        p = x - L_T( u_1 )
        if  np.sum ( (u_1 - u)**2 )  < tol and n_iter > 1:
            #print('prox_general_frame converged after ' +str(n_iter) + ' iterations.')
            converged = True
        u = np.copy(u_1)
        if n_iter >= max_iter:
            #print('prox_general_frame did not converge.')
            break
    return p

def positivity_general_frame(x,L,L_T,tol,max_iter=5):
    if tol == 0:
        tol = 0.1
    n_iter = 0
    u = 0#np.copy(L(x))
    p = np.copy(x)
    converged = False
    while not converged:
        n_iter += 1
        temp = u + L(p)
        u_1 = ( temp -  positivity(temp) )
        p = x - L_T( u_1 )
        if  np.sum ( (u_1 - u)**2 )  < tol and n_iter > 1:
            converged = True
        u = np.copy(u_1)
        if n_iter >= max_iter:
            break
    return p

def prox_tight_frame(x,_lambda,prox_operator,L,L_T,nu):
    L_x = L(x)
    return x + L_T( prox_operator(L_x,_lambda) - L_x )/nu

def proj_tight_frame(x,proj_operator,L,L_T,nu):
    L_x = L(x)
    return x + L_T( proj_operator(L_x) - L_x )/nu



def A(f,K):
    return np.real(np.fft.ifft2(np.fft.fft2(f) * K))
def A_(f,K):
    return np.real(np.fft.ifft2(np.fft.fft2(f) * np.ma.conjugate(K)))

def soft_thresh(U,param):
    return (np.abs(U)>param)*(U-sign(U)*param)

def sign(U):
    return 1*(U>0) - 1*(U<0)



# define the operator W as in paper, uses db4 wavelets
def W(coeffs_array):
    coeffs_from_arr = pywt.array_to_coeffs(coeffs_array, coeff_slices_GLOBAL)
    return pywt.waverecn(coeffs_from_arr, wavelet=wavelet_type_GLOBAL)

# define inverse of operator W as in paper, uses db4 wavelets
def W_(frame):
    coeffs = pywt.wavedec2(frame, wavelet_type_GLOBAL)
    coeffs_array,_ = pywt.coeffs_to_array(coeffs)
    return coeffs_array

def wavelet_frame_thresh(frame,fraction_coeff):
    n,_ = frame.shape
    frame_wt = W_(frame)
    sorted_wt = np.sort(np.ravel(abs(frame_wt)))[::-1]
    n_pix, = sorted_wt.shape
    treshold_value = sorted_wt[int(n_pix*fraction_coeff)]
    frame_wt_T = np.multiply(frame_wt,(abs(frame_wt) > treshold_value))
    frame_T = W(frame_wt_T)[:n,:n] #checker si c'est ok de faire comme ca.??
    return frame_T

'''
proj_low_rank
'''
def proj_low_rank(x,k):
    #x=x*1.
    U, s, V = randomized_svd(x, n_components=k, n_iter=5,transpose='auto')
    #U, s, V = np.linalg.svd(x,full_matrices=False)
    s[k:] = 0
    return np.dot(U,np.dot(np.diag(s),V))




def frame_euclidean_proj_l1ball(frame,_lambda):
    if _lambda==0:
        return frame*0
    else:
        return euclidean_proj_l1ball(frame.ravel(),_lambda).reshape(frame.shape)

def proj_l2_ball(x,_lambda):
    if _lambda==0:
        return x*0
    else:
        l2_norm = np.sqrt( np.sum( x**2 ) )
        if l2_norm > _lambda:
            return _lambda * x / l2_norm
        else:
            return x

def prox_translation(x,y,prox_operator):
    return y + prox_operator(x - y)

'''
proj_l0_ball
'''
def proj_l0_ball(x,k):
    m,n = x.shape
    x_temp = np.copy(x).reshape(m*n)
    x_temp = x_temp * (x_temp>0)
    ind = np.argsort(-np.abs(x_temp))
    tau = np.abs(x_temp[ind][k])
    x_temp = x_temp.reshape(m,n)
    x_temp[np.abs(x)<tau] = 0
    return x_temp

def shearlet_frame_thresh(frame,fraction_coeff):
    n,_ = frame.shape
    frame_wt = pyshearlab.SLsheardec2D(frame, shearletSystem)
    sorted_wt = np.sort(np.ravel(abs(frame_wt)))[::-1]
    n_pix, = sorted_wt.shape
    treshold_value = sorted_wt[int(n_pix*fraction_coeff)]
    frame_wt_T = np.multiply(frame_wt,(abs(frame_wt) > treshold_value))
    frame_T = pyshearlab.SLshearrec2D(frame_wt_T, shearletSystem)
    return frame_T



def disk_to_cube(array,angles,interpolation):
    n,_ = array.shape
    t, = angles.shape
    cube = np.outer(np.ones([t]),array).reshape(t,n,n)
    cube = vip.preproc.cube_derotate(cube,-angles,interpolation=interpolation)
    return cube

def cube_to_frame(cube,angles,interpolation):
    cube = vip.preproc.cube_derotate(cube,angles,interpolation=interpolation)
    # below: sum and not mean since we deal with p^T (from paper)
    array = np.sum(cube,axis=0)
    return array

def compute_cube_frame_grad(xd,xp,xl,matrix,angles,interpolation,**kwargs):
    t,_ = xl.shape
    xs = xd+xp
    n,_ = xs.shape
    grad_l = xl + disk_to_cube(xs,angles,interpolation).reshape(t,n*n) - matrix
    grad_s = t*xs + cube_to_frame((xl - matrix).reshape(t,n,n),angles,interpolation)
    #grad_s = xs + cube_to_frame((xl - matrix).reshape(t,n,n),angles,interpolation)/t
    obj = l2_convergence(xd,xp,xl,matrix,angles,interpolation,**kwargs)
    return grad_s,grad_s, grad_l,obj

def compute_cube_frame_conv_grad(xd,xp,xl,matrix,angles,interpolation,kernel=1,mask_center=0,**kwargs):
    t,_ = xl.shape
    xs = xd+xp
    n,_ = xs.shape
    mask = get_donut(np.ones((n,n)),center_image,mask_center)
    cube_L_minus_matrix = (xl - matrix).reshape(t,n,n)
    grad_l = (mask*(disk_to_cube(A(xs,kernel),angles,interpolation) +cube_L_minus_matrix)).reshape(t,n*n)
    # not sure about the 1./t, it is part of the mu in fact...
    temp = mask*(t*A(xs,kernel) + cube_to_frame(cube_L_minus_matrix,angles,interpolation))
    grad_s = A_(temp,kernel)
    #grad_s = 1./t*A_(temp,kernel)
    obj = mayo_hci.l2_convergence_convolution(xl,xs,matrix,angles,interpolation,kernel=kernel,**kwargs)
    return grad_s, grad_s, grad_l ,obj

def positivity_mask_center(array,r_mask):
    n,_ = array.shape
    return positivity(vip.var.get_circle(vip.var.mask_circle(array,r_mask),n/2))

def positivity(array):
    return array*(array>0)

def compute_cube_frame_conv_grad_huber_loss(xd,xp,xl,matrix,angles,interpolation,kernel=1,huber_param=1,mask_center=0,**kwargs):
    t,_ = xl.shape
    xs = xd+xp
    n,_ = xs.shape
    center_mask = vip.var.get_circle(vip.var.mask_circle(np.ones((n,n)),mask_center),n/2)
    reference_array = xl.reshape(t,n,n) + disk_to_cube(A(xs,kernel),angles,interpolation) - matrix.reshape(t,n,n)
    huber_mask = np.abs(reference_array) < huber_param
    #huber_mix_array = mask*reference_array/huber_param + sign((1.-mask)*reference_array)
    huber_mix_array = huber_mask*reference_array + sign((1.-huber_mask)*reference_array)*huber_param
    grad_l = (center_mask*huber_mix_array).reshape(t,n*n)
    temp = center_mask*cube_to_frame(huber_mix_array,angles,interpolation)
    grad_s = A_(temp,kernel)
    obj = mayo_hci.l2_convergence_convolution(xl,xs,matrix,angles,interpolation,kernel=kernel,**kwargs)
    return  grad_s, grad_s, grad_l, obj

def compute_cube_frame_conv_grad_huber_loss_synthesis(x_disk,x_planet,xl,matrix,angles,interpolation,L,L_T,kernel,huber_param,mask_center):
    t,_ = xl.shape
    n,_ = x_planet.shape
    center_mask = vip.var.get_circle(vip.var.mask_circle(np.ones((n,n)),mask_center),n/2)
    reference_array = xl.reshape(t,n,n) + disk_to_cube(A(x_planet + L(x_disk),kernel),angles,interpolation) - matrix.reshape(t,n,n)
    huber_mask = np.abs(reference_array) < huber_param
    #huber_mix_array = mask*reference_array/huber_param + sign((1.-mask)*reference_array)
    huber_mix_array = huber_mask*reference_array + sign((1.-huber_mask)*reference_array)*huber_param
    grad_l = (center_mask*huber_mix_array).reshape(t,n*n)
    temp = cube_to_frame(center_mask*huber_mix_array,angles,interpolation)
    grad_planet = A_(temp,kernel)
    grad_disk = L_T(A_(temp,kernel))
    return grad_disk, grad_planet, grad_l

def compute_cube_frame_grad(x_disk,x_planet,xl,matrix,angles,interpolation,L,L_T):
    t,_ = xl.shape
    n,_ = x_planet.shape
    grad_l = xl + disk_to_cube(x_planet + L(x_disk),angles,interpolation).reshape(t,n*n) - matrix
    grad_planet = t*(x_planet+L(x_disk))  + cube_to_frame((xl - matrix).reshape(t,n,n),angles,interpolation)
    grad_disk = L_T(grad_planet)
    #grad_s = xs + cube_to_frame((xl - matrix).reshape(t,n,n),angles,interpolation)/t
    return grad_l ,grad_disk, grad_planet



def construct_list_full_rotation_matrix(angles,t,n):
    list_full_rotation_matrix = []
    frame_circle = vip.var.get_circle(np.ones((n,n)),n/2-3)
    for k in range(t):
        full_rotation_matrix = dok_matrix((n**2,n**2), dtype=np.float32)
        ang_rad = angles[k]*np.pi/180
        rot_matrix = np.array([ [np.cos(ang_rad), - np.sin(ang_rad)], [np.sin(ang_rad), np.cos(ang_rad)] ])
        for i in range(n):
            for j in range(n):
                if frame_circle[i,j] == 1:
                    coord = np.array([i-n/2+1,j-n/2+1])
                    coord = np.array([i-n/2+1,j-n/2])
                    continuous_rot_coord = rot_matrix @ coord
                    floored_rot_coord = np.rint(continuous_rot_coord).astype(int)
                    floored_rot_coord += int(n/2)-1
                    #continuous_rot_coord - floored_rot_coord
                    full_rotation_matrix[i*n+j, floored_rot_coord[0]*n + floored_rot_coord[1] ] = 1
        list_full_rotation_matrix.append(full_rotation_matrix) 
        #print(type(list_full_rotation_matrix[k]) )
        del full_rotation_matrix
    return list_full_rotation_matrix
    
def scipy_matrix_derote(matrix,list_full_rotation_matrix):
    t,N = matrix.shape
    derot_matrix = np.zeros((t,N))
    for k in range(t):
        derot_matrix[k,:] = list_full_rotation_matrix[k].dot(matrix[k,:])
        print(type(list_full_rotation_matrix[k]))
    return derot_matrix

def scipy_matrix_adjoint_derote(matrix,list_full_rotation_matrix):
    t,N = matrix.shape
    ajd_derot_matrix = np.zeros((t,N))
    for k in range(t):
        ajd_derot_matrix[k,:] = list_full_rotation_matrix[k].T.dot(matrix[k,:])
    return ajd_derot_matrix

import time

def disk_to_matrix_scipy_matrix_derote(frame,list_full_rotation_matrix):
    t = len(list_full_rotation_matrix)
    matrix = np.outer(np.ones( (t) ),frame)
    return scipy_matrix_derote(matrix,list_full_rotation_matrix)

def adj_disk_to_matrix_scipy_matrix_derote(matrix,list_full_rotation_matrix):
    matrix = scipy_matrix_adjoint_derote(matrix,list_full_rotation_matrix)
    return np.sum(matrix,axis=0)

# This is a numpy code used to fit the residuals to the huber-loss
def compute_huber_loss(x,huber_delta,a):
    abs_x = np.abs(x)
    c2 = 1
    return np.where(abs_x < huber_delta, a * abs_x ** 2, 2*a*huber_delta*abs_x -a*huber_delta**2)

def compute_normalized_huber_loss_alternate_def(x,normalized_huber_delta):
    abs_x = torch.abs(x)
    return torch.where(abs_x < normalized_huber_delta, 0.5 * abs_x ** 2, normalized_huber_delta*abs_x - normalized_huber_delta**2/2).sum()

# This is the torch code used in the MAYO algorithm itself
def compute_normalized_huber_loss(x,huber_delta,Xi):
    abs_x = torch.abs(x/Xi)
    return torch.where(abs_x < huber_delta, Xi*0.5 * abs_x ** 2, Xi*huber_delta*abs_x - huber_delta**2/2).sum()


#not used
def compute_generalized_huber_loss(x,huber_param,c1,c2):
    abs_x = torch.abs(x)
    return torch.where(abs_x < huber_param, 0.5 * abs_x ** 2, 0.5*huber_param/c1 * (c2*abs_x +(c1-c2)*huber_param ) ).sum()


def compute_l2_loss(x):
    return 0.5*(x**2).sum()

def compute_cube_frame_conv_only_planet_grad_pytorch(xp,xl,matrix,angles,compute_loss,kernel,mask, center_image):
    grad_d_p, _, np_grad_L, loss = compute_cube_frame_conv_grad_pytorch(xp,xp*0,xl,matrix,angles,compute_loss,kernel,mask, center_image)
    return grad_d_p, np_grad_L, loss

def compute_MAYO_grad_stochastic(xd,xp,xl,matrix,angles,compute_loss,kernel,mask, center_image,proportion_frames):
    t,_ = matrix.shape
    if t != angles.shape[0]:
        print('ANGLES do not have t elements!')
    nframes = int(t*proportion_frames)
    if nframes <1:
        print('Not enough frames !! Setting n_frames to 1')
        nframes = 1
    elif nframes > t:
        print('Too many frames, etting n_frames to int(3*t/4)')
        nframes = int(3*t/4)
    xs = xd + xp
    n,_ = xs.shape
    conv_op = lambda x : A(x,kernel)
    adj_conv_op = lambda x : A_(x,kernel)
    
    class FFTconv_numpy_torch(torch.autograd.Function):
        @staticmethod
        def forward(ctx, input):
            numpy_input = input.detach().numpy()
            result = conv_op(numpy_input)
            return input.new(result)
        @staticmethod
        def backward(ctx, grad_output):
            numpy_go = grad_output.numpy()
            result = adj_conv_op(numpy_go)
            return grad_output.new(result)
    def fft_conv_np_torch(input):
        return FFTconv_numpy_torch.apply(input)

    # needed for rotation:
    center: torch.tensor = torch.ones(1, 2)
    center[..., 0] = center_image[0] # x
    center[..., 1] = center_image[1] # y
    scale: torch.tensor = torch.ones(1)

    temp_t = np.arange(t)
    np.random.shuffle(temp_t)
    index_grad_s = temp_t[:nframes]
    index_no_grad_s = temp_t[nframes:]


    torch_xs = torch.tensor([[xs]],requires_grad=True)


    torch_data = torch.tensor(matrix.reshape(t,n,n))
    torch_L = torch.tensor(xl.reshape(t,n,n)[index_grad_s,:,:],requires_grad=True)
    

    loss : torch.tensor = torch.zeros(1,requires_grad=True)
    for index_L,k in enumerate(index_grad_s):
        angle: torch.tensor = torch.ones(1) * (angles[k])
        M: torch.tensor = kornia.get_rotation_matrix2d(center, angle, scale)
        rotated_xs = kornia.warp_affine(torch_xs.float(), M, dsize=(n,n))
        #xs_rotated[k,:,:] = rotated_xs.detach().numpy()
        loss = loss + compute_loss( fft_conv_np_torch(rotated_xs[0,0,:,:]) + torch_L[index_L,:,:] - torch_data[k,:,:]) 
    loss.backward()
    torch_grad_xs = torch_xs.grad
    np_grad_xs = torch_grad_xs[0,0,:,:].detach().numpy()
    grad_d_p = np_grad_xs*mask
    torch_grad_L__index_grad_s = torch_L.grad.detach().numpy()

    torch_L = torch.tensor(xl.reshape(t,n,n)[index_no_grad_s,:,:],requires_grad=True)
    torch_xs = torch.tensor([[xs]],requires_grad=False)     # requires_grad = False because at this point, we don't need the gradient of xs, it is already done
    loss : torch.tensor = torch.zeros(1,requires_grad=True)
    for index_L,k in enumerate(index_grad_s):
        angle: torch.tensor = torch.ones(1) * (angles[k])
        M: torch.tensor = kornia.get_rotation_matrix2d(center, angle, scale)
        rotated_xs = kornia.warp_affine(torch_xs.float(), M, dsize=(n,n))
        #xs_rotated[k,:,:] = rotated_xs.detach().numpy()
        loss = loss + compute_loss( fft_conv_np_torch(rotated_xs[0,0,:,:]) + torch_L[index_L,:,:] - torch_data[k,:,:]) 
    loss.backward()
    torch_grad_L__index_no_grad_s = torch_L.grad.detach().numpy()

    #np_grad_L = torch_grad_L.detach().numpy()*mask
    np_grad_L = np.zeros((t,n,n))
    np_grad_L[index_grad_s,:,:] = np.copy(torch_grad_L__index_grad_s)
    np_grad_L[index_no_grad_s,:,:] = np.copy(torch_grad_L__index_no_grad_s)
    #grad_d_p = A_(np_grad_xs,kernel)*mask
    return grad_d_p, grad_d_p, np_grad_L.reshape(t,n*n), loss.detach().numpy()

def compute_cube_frame_conv_grad_pytorch(xd,xp,xl,matrix,angles,compute_loss,kernel,mask, center_image):
    t,_ = matrix.shape
    xs = xd + xp
    n,_ = xs.shape
    conv_op = lambda x : A(x,kernel)
    adj_conv_op = lambda x : A_(x,kernel)
    if t != angles.shape[0]:
        print('ANGLES do not have t elements!')
    
    class FFTconv_numpy_torch(torch.autograd.Function):
        @staticmethod
        def forward(ctx, input):
            numpy_input = input.detach().numpy()
            result = conv_op(numpy_input)
            return input.new(result)
        @staticmethod
        def backward(ctx, grad_output):
            numpy_go = grad_output.numpy()
            result = adj_conv_op(numpy_go)
            return grad_output.new(result)
    def fft_conv_np_torch(input):
        return FFTconv_numpy_torch.apply(input)

    # needed for rotation:
    center: torch.tensor = torch.ones(1, 2)
    center[..., 0] = center_image[0] # x
    center[..., 1] = center_image[1] # y
    scale: torch.tensor = torch.ones(1)

    torch_xs = torch.tensor([[xs]],requires_grad=True)


    torch_data = torch.tensor(matrix.reshape(t,n,n))
    torch_L = torch.tensor(xl.reshape(t,n,n),requires_grad=True)
    

    loss : torch.tensor = torch.zeros(1,requires_grad=True)

    for k in range(t):
        angle: torch.tensor = torch.ones(1) * (angles[k])
        M: torch.tensor = kornia.get_rotation_matrix2d(center, angle, scale)
        rotated_xs = kornia.warp_affine(torch_xs.float(), M, dsize=(n,n))
        #xs_rotated[k,:,:] = rotated_xs.detach().numpy()
        loss = loss + compute_loss( fft_conv_np_torch(rotated_xs[0,0,:,:]) + torch_L[k,:,:] - torch_data[k,:,:]) 
    loss.backward()
    torch_grad_xs = torch_xs.grad
    torch_grad_L = torch_L.grad


    np_grad_xs = torch_grad_xs[0,0,:,:].detach().numpy()
    #np_grad_L = torch_grad_L.detach().numpy()*mask
    np_grad_L = torch_grad_L.detach().numpy()
    #grad_d_p = A_(np_grad_xs,kernel)*mask
    grad_d_p = np_grad_xs*mask
    return grad_d_p, grad_d_p, np_grad_L.reshape(t,n*n), loss.detach().numpy()

def compute_rotated_cube_grad_pytorch(xd,xp,bar_data,compute_loss,kernel,mask):
    t,_,_ = bar_data.shape
        
    xs = A(xd+xp,kernel)
    n,_ = xs.shape

    torch_bar_data = torch.tensor(bar_data)
    torch_xs = torch.tensor(xs,requires_grad=True)

    loss : torch.tensor = torch.zeros(1,requires_grad=True)

    for k in range(t):
        loss = loss + compute_loss(torch_xs - torch_bar_data[k,:,:]) 
    loss.backward()
    torch_grad_xs = torch_xs.grad


    np_grad_xs = torch_grad_xs.detach().numpy()
    grad_d_p = A_(np_grad_xs,kernel)*mask
    return grad_d_p, grad_d_p, loss.detach().numpy()

def grad_MCA_pytorch(xd,xp,noisy_disk_planet,compute_loss,conv_op,adj_conv_op,mask):
    xs = xd + xp
    n,_ = xs.shape

    class FFTconv_numpy_torch(torch.autograd.Function):
        @staticmethod
        def forward(ctx, input):
            numpy_input = input.detach().numpy()
            result = conv_op(numpy_input)
            return input.new(result)
        @staticmethod
        def backward(ctx, grad_output):
            numpy_go = grad_output.numpy()
            result = adj_conv_op(numpy_go)
            return grad_output.new(result)
    def fft_conv_np_torch(input):
        return FFTconv_numpy_torch.apply(input)
    
    torch_data = torch.tensor(noisy_disk_planet)
    torch_xs = torch.tensor(xs,requires_grad=True)

    loss = compute_loss(fft_conv_np_torch(torch_xs) - torch_data)
    loss.backward()
    torch_grad_xs = torch_xs.grad



    np_grad_xs = torch_grad_xs.detach().numpy()
    #return A_(np_grad_xs,kernel)*mask , loss.detach().numpy()
    grad_xs = np_grad_xs*mask
    return  grad_xs, grad_xs, loss.detach().numpy()


def grad_MCA_pytorch_bckp(xd,xp,noisy_disk_planet,compute_loss,conv_op,adj_conv_op,mask_center):
    xs = conv_op(xd+xp)
    n,_ = xs.shape

    torch_data = torch.tensor(noisy_disk_planet)
    torch_xs = torch.tensor(xs,requires_grad=True)
    

    loss = compute_loss(torch_xs - torch_data) 
    loss.backward()
    torch_grad_xs = torch_xs.grad

    mask = get_donut(np.ones((n,n)),center_image,mask_center)

    np_grad_xs = torch_grad_xs.detach().numpy()

    #return A_(np_grad_xs,kernel)*mask , loss.detach().numpy()
    grad_xs = adj_conv_op(np_grad_xs)*mask
    return  grad_xs,grad_xs, loss.detach().numpy()


def compute_cube_frame_conv_grad_pytorch_l2_shearlets_regul(xd,xp,xl,w,matrix,angles,compute_loss,kernel,mask, center_image,l2_shear_regul_param,Phi,Phi_T):
    grad_d, grad_p, grad_L, loss = compute_cube_frame_conv_grad_pytorch(xd,xp,xl,matrix,angles,compute_loss,kernel,mask, center_image)
    temp = w - Phi(xd)
    grad_w = l2_shear_regul_param * temp
    grad_d += - l2_shear_regul_param * Phi_T(temp)
    loss += l2_shear_regul_param/2*np.sum( temp**2 )
    return grad_d, grad_p, grad_L, grad_w, loss

def compute_cube_frame_conv_grad_pytorch_shearlets_synthesis(xd,xp,xl,w,matrix,angles,compute_loss,kernel,mask, center_image,l2_shear_regul_param,Phi,Phi_T):
    grad_disk_comp, grad_p, grad_L, loss = compute_cube_frame_conv_grad_pytorch( (Phi(xd)+w)/2,xp,xl,matrix,angles,compute_loss,kernel,mask, center_image)
    temp = w - Phi(xd)
    grad_d = Phi_T(grad_disk_comp - l2_shear_regul_param * temp)
    grad_w = grad_disk_comp + l2_shear_regul_param * temp
    loss += l2_shear_regul_param/2*np.sum( temp**2 )
    return grad_d, grad_p, grad_L, grad_w, loss




def compute_cube_frame_grad_pytorch_no_regul(xs,xl,matrix,angles,compute_loss, mask, center_image):
    t,_ = matrix.shape
    n,_ = xs.shape


    # needed for rotation:
    center: torch.tensor = torch.ones(1, 2)
    center[..., 0] = center_image[0] # x
    center[..., 1] = center_image[1] # y
    scale: torch.tensor = torch.ones(1)

    torch_xs = torch.tensor([[xs]],requires_grad=True)

    torch_data = torch.tensor(matrix.reshape(t,n,n))
    torch_L = torch.tensor(xl.reshape(t,n,n),requires_grad=True)

    loss : torch.tensor = torch.zeros(1,requires_grad=True)

    for k in range(t):
        angle: torch.tensor = torch.ones(1) * (angles[k])
        M: torch.tensor = kornia.get_rotation_matrix2d(center, angle, scale)
        rotated_xs = kornia.warp_affine(torch_xs.float(), M, dsize=(n,n))
        loss = loss + compute_loss( (rotated_xs[0,0,:,:]) + torch_L[k,:,:] - torch_data[k,:,:]) 
    loss.backward()
    torch_grad_xs = torch_xs.grad
    torch_grad_L = torch_L.grad

    np_grad_xs = torch_grad_xs[0,0,:,:].detach().numpy()
    np_grad_L = torch_grad_L.detach().numpy()*mask
    return np_grad_xs*mask, np_grad_L.reshape(t,n*n), loss.detach().numpy()


def cube_rotate_kornia(cube,angles,center_image):
    t,n,_ = cube.shape
    if t != angles.shape[0]:
        print('ANGLES do not have t elements!')
    center: torch.tensor = torch.ones(1, 2)
    center[..., 0] = center_image[0] # x
    center[..., 1] = center_image[1] # y
    scale: torch.tensor = torch.ones(1)

    cube_rotated = np.zeros((t,n,n))

    for k in range(t):
        torch_xs = torch.tensor([[cube[k,:,:]]],requires_grad=False)
        angle: torch.tensor = torch.ones(1) * (angles[k])
        M: torch.tensor = kornia.get_rotation_matrix2d(center, angle, scale)
        cube_rotated[k,:,:] = kornia.warp_affine(torch_xs.float(), M, dsize=(n,n)).numpy()
    return cube_rotated


def get_rotation_center_and_mask(n,corono_radius,center_coord):
    if center_coord:
        pass
    else:
        center_coord = (n // 2 - 0.5, n // 2 - 0.5)
    cx, cy = center_coord
    xx, yy = np.ogrid[:n, :n]
    circle =  np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2) # eq of circle. sq dist to center
    inner_circle_mask = (circle > corono_radius)  # boolean mask
    outside_circle_mask = (circle < np.floor(n/2-4))
    mask = inner_circle_mask*outside_circle_mask
    return center_coord, mask


def get_huber_parameters(algo):
    print('Estimating the Huber-loss parameters...')
    N_bins = 200
    width_annulus = 5
    proportion_pixels_to_keep = 0.999841
    proportion_pixels_to_keep = 0.999
    if algo.parameters_algo['data_name'] == "SPHERE_HR4796A_clean":
        proportion_pixels_to_keep = 0.99
    t,n,_ = algo.data.shape
    U_L0,_,_ = randomized_svd(algo.xl.reshape(t,n*n), n_components=algo.parameters_algo['rank'], n_iter=5,transpose='auto')
    Low_rank_xl = (U_L0 @ U_L0.T @ algo.xl.reshape(t,n*n) ).reshape(t,n,n)
    cube_xs = np.zeros((algo.t,algo.n,algo.n))
    cube_xs[:,:,:] = algo.GreeDS_frame 
    cube_xs = cube_rotate_kornia(cube_xs,algo.angles,algo.center_image)
    error = algo.data - Low_rank_xl - cube_xs

    sigma_by_annulus = np.ones((n,n))
    for i in range(n//(width_annulus*2)):
        inner_annulus = i*width_annulus
        annulus_mask_ind = vip.var.shapes.get_annulus_segments(sigma_by_annulus,inner_annulus,width_annulus,mode='ind')
        errors_in_annulus = error[:,annulus_mask_ind[0][0],annulus_mask_ind[0][1]]
        sigma_by_annulus[annulus_mask_ind[0][0],annulus_mask_ind[0][1]] = np.sqrt(np.var(errors_in_annulus))

    normalized_error = error/sigma_by_annulus*algo.mask

    min_values = 0
    max_values = np.sort(np.abs(normalized_error).ravel())[int(proportion_pixels_to_keep*t*n**2)] # 

    pdf = np.zeros(N_bins)*0.
    values = np.linspace(min_values,max_values,N_bins+1)

    total_number_in_bin = 0
    for kk in range(N_bins):
        #INV_sub_exp_levels[kk] = np.sum(error>=values[kk])*1.
        who_is_in_bin = (np.abs(normalized_error) >= values[kk])*1.*(np.abs(normalized_error) < values[kk+1])*algo.mask
        pdf[kk] = (np.sum(who_is_in_bin))
        total_number_in_bin += pdf[kk]
    pdf = pdf/total_number_in_bin
    #INV_sub_exp_levels = INV_sub_exp_levels/n**2
    values = values[:-1]

    y = pdf*(pdf>0)
    #y += 10**-10
    bins_to_consider = np.where(y>0) # we do not look at bins with zero elements
    y = y[bins_to_consider]
    y /= np.max(y)

    y = -np.log(y)

    x = values[bins_to_consider]
    #import matplotlib.pyplot as plt
    #plt.figure()
    from scipy.optimize import curve_fit
    try:
        pw, cov = curve_fit(compute_huber_loss, x, y)
        print('Huber-loss parameters successfully estimated')
    except:
        print('Huber-loss NOT ESTIMATED!')
        import sys
        if sys.platform != 'linux': # for me, this means I am running on keneda or nielsen
            import matplotlib.pyplot as plt
            plt.plot(x, y, 'o')
            plt.show()
            print('We display the negative log-likelihood of normalized residuals')
        else:
            print('No display available, run in local to look at the negative log-likelihood of normalized residuals')
        c = float(input('Choose value of c :   ') )
        delta = float(input('Choose value of delta :   ') )
        pw = delta, c
    #    pw = (0,0)
    algo.negative_log_hist_x = x
    algo.negative_log_hist_y = y
    algo.fitted_pw = pw
    print(pw)
    #plt.plot(x, y, 'o', x, compute_huber_loss(x, *pw), 'r-')
    #plt.plot( (pw[0],pw[0]), (0, np.max(y)),'g')
    #plt.show()
    return pw, sigma_by_annulus