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
import numpy as np

import json
import sys

from utils import compute_stim_map
from hciplot import plot_frames as plots


def automatic_load_data(data_name,channel=0,dir='./',RDI=False,quick_look=0,crop=0):

    with open(dir+data_name+'/0_import_info.json', 'r') as read_file_import_info:
        import_info = json.load(read_file_import_info)
    
    if import_info['cube']:
        data = vip.fits.open_fits(dir+data_name+'/'+import_info['cube'][channel]['location'])
        if len(data.shape) == 4:
            data = data[channel,:,:,:]
        t,n_init,_ = data.shape
        if crop:
            if n_init > crop:
                data = vip.preproc.cosmetics.cube_crop_frames(data, crop, xy=(n_init//2+1,n_init//2+1),force=True)
            else:
                print('crop is larger than initial image, no cropped performed')
        _,n,_ = data.shape
        print('imported cube on channel: ' + import_info['cube'][channel]['channel_name'])
    else:
        cube = None
        print('data entry missing in import info')

    if import_info['angles']:
        angles = vip.fits.open_fits(dir+data_name+'/'+import_info['angles'][channel]['location'])
        if len(angles.shape) == 2:
            angles = angles[channel,:]
        angles = angles*import_info['angles_multiply']
        if angles.shape[0] != t:
            print('not right numbers of angles')
    else:
        angles = None
        print('angles entry missing in import info')

    if import_info['psf']:
        psf = vip.fits.open_fits(dir+data_name+'/'+import_info['psf'][channel]['location'])
        if len(psf.shape) == 4:
            try:
                psf = psf[channel,import_info['psf']['which_psf'],:,:]
            except:
                psf = psf[channel,0,:,:]
        elif len(psf.shape) == 3:
            print('we are here')
            psf = psf[channel,:,:]
        print(psf.shape)
        n_psf,_ = psf.shape
        print(psf.shape)
        psf = psf*(psf>0)
        #psf -= np.min(psf)
        if n_psf < n:
            print('dimension of psf modified to fit data')
            psf_width = psf.shape[0]
            ind_inf = int(n/2-psf_width/2)
            ind_sup = int(n/2+psf_width/2)
            psf_full = np.zeros((n,n))
            psf_full[ind_inf:ind_sup,ind_inf:ind_sup] = psf
            psf = psf_full
        if n_psf > n:
            psf = vip.preproc.cosmetics.frame_crop(psf, crop,force=True)
        psf = psf/np.sum(np.abs(psf))
        print('psf normalized')
    else:
        psf = None
        print('psf entry missing in import info')
    if RDI:
        RDI_cube = vip.fits.open_fits(dir+data_name+'/'+import_info['RDI_cube'][channel]['location'])
        n_init_crop,_,_ = RDI_cube.shape
        print('we are in RDI from automatic load')
        print(n_init_crop)
        print(crop)
        if crop:
            if n_init_crop > crop:
                RDI_cube = vip.preproc.cosmetics.cube_crop_frames(RDI_cube, crop, xy=(n_init//2+1,n_init//2+1),force=True)
            else:
                print('crop is larger than initial image, no cropped performed')
        return data,angles,psf, RDI_cube
    if quick_look:
        frame, _, _, residuals_cube, _ = vip.pca.pca_fullfr.pca(data,angles,ncomp=quick_look,full_output=True)
        stim_frame = compute_stim_map(vip.preproc.cube_derotate(residuals_cube,angles)) 
        plots((frame,stim_frame))
        return data,angles,psf,frame,stim_frame
    else:
        return data,angles,psf
