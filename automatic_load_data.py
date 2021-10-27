'''
automatic_load_data deals with loading ADI data 
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
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  Sfee the
    GNU General Public License for more details.
'''


import torch


import json
import sys
import os


import mayo_hci


def automatic_load_data(data_name,device,channel=0,data_dir='default',crop=0):
    """
    automatic_load_data(data_name,channel=0,data_dir='default',RDI=False,quick_look=0,crop=0,center_im=None)
        loads ADI datasets automatically 
    Parameters
    ----------
    data_name : str
        name of the data to use in MAYO
    channel : int
        channel to use if multiple channels, default is 0
    crop : int
        if crop > 0, crops the ADI cube to the desired width, default is 0
    
    Returns
    -------
    data : numpy array
        t x n x n ADI dataset
    angles :  numpy array
        list of angles
    psf :  numpy array
        n x n psf
    center_im : tuple
        (x,y), center of the frame    
    """ 
    if data_dir == 'default':
        try:
            with open(os.path.dirname(mayo_hci.__file__) + '/data_path.json', 'r') as read_data_path:
                temp = json.load(read_data_path)
                data_dir = temp['default_path_to_data']
        except FileNotFoundError:
            data_dir = './'
    with open(data_dir+data_name+'/0_import_info.json', 'r') as read_file_import_info:
        import_info = json.load(read_file_import_info)
    center = import_info['center_image']
    if import_info['cube']:
        data = mayo_hci.open_fits(data_dir+data_name+'/'+import_info['cube'][channel]['location'],device)
        if len(data.shape) == 4:
            data = data[channel,:,:,:]
        t,n_init,_ = data.shape
        if crop:
            if n_init > crop:
                n = crop
                data = data[:,int(center[0])-n//2:int(center[0])+n//2,int(center[1])-n//2:int(center[1])+n//2]
                center_im = (n//2 + (center[0]-int(center[0])),n//2 + (center[1]-int(center[1])))
            else:
                print('crop is larger than initial image, no cropped performed')
                _,n,_ = data.shape
        else:
            n = n_init
            center_im = center
        print('imported cube on channel: ' + import_info['cube'][channel]['channel_name'])
    else:
        cube = None
        print('data entry missing in import info')

    if import_info['angles']:
        angles = mayo_hci.open_fits(data_dir+data_name+'/'+import_info['angles'][channel]['location'],device)
        if len(angles.shape) == 2:
            angles = angles[channel,:]
        angles = angles*import_info['angles_multiply']
        if angles.shape[0] != t:
            print('not right numbers of angles')
    else:
        angles = None
        print('angles entry missing in import info')
            
    if import_info['psf']:
        psf = mayo_hci.open_fits(data_dir+data_name+'/'+import_info['psf'][channel]['location'],device)
        if len(psf.shape) == 4:
            try:
                psf = psf[channel,import_info['psf']['which_psf'],:,:]
            except:
                psf = psf[channel,0,:,:]
        elif len(psf.shape) == 3:
            psf = psf[channel,:,:]
        n_psf,_ = psf.shape
        if n_psf != n_init:
            print('Warning, size of psf is not the same as data')
        psf = psf*(psf>0)
        if n_psf < n:
            psf_full = torch.zeros(n,n,device=device)
            if n_psf % 2==0:
                psf_full[n//2-n_psf//2:n//2+n_psf//2,n//2-n_psf//2:n//2+n_psf//2] = psf
            else:
                psf_full[n//2-n_psf//2:n//2+n_psf//2+1,n//2-n_psf//2:n//2+n_psf//2+1] = psf
            psf = psf_full
        if n_psf > n:
            psf = psf[n_psf//2-n//2:n_psf//2+n//2,n_psf//2-n//2:n_psf//2+n//2]
        psf = psf/torch.abs(psf).sum()
        print('psf normalized')
    else:
        psf = None
        print('psf entry missing in import info')
    if data.max()<0.1:
        data *= 10**4
        print('data assumed in contrast units, multiplied by 10**4')
    return data,angles,psf,center_im
