'''
Injects synthetic circumstellar signal into an empty ADI dataset
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
import numpy as np

import mayo_hci
import torch
import kornia

def create_synthetic_data_with_disk_planet(empty_data,angles,psf,add_synthetic_signal):
    """
    automatic_load_data(data_name,channel=0,dir='default',RDI=False,quick_look=0,crop=0,center_im=None)
        loads ADI datasets automatically 
    Parameters
    ----------
    data : numpy array
        t x n x n the empty (without circumstellar signal) ADI dataset
    angles :  numpy array
        list of angles
    psf :  numpy array
        n x n psf
    add_synthetic_signal
        tuple containing all the properties of the injected signal 
    Returns
    -------
    data : numpy array
        t x n x n, ADI dataset (empty_data + circumstellar signal injected)
    disk : numpy array
        n x n, the injected circumstellar signal (disk+planet) 
    """ 
    disk_max_intensity = add_synthetic_signal['disk_max_intensity']
    disk_inclination = add_synthetic_signal['disk_inclination']
    planets_positions_intensities = tuple(add_synthetic_signal['planets_positions_intensities'])
    if 'xdo' not in add_synthetic_signal:
        add_synthetic_signal['xdo'] = 0
    if 'ydo' not in add_synthetic_signal:
        add_synthetic_signal['ydo'] = 0
    if 'pa' not in add_synthetic_signal:
        add_synthetic_signal['pa'] = 80
    if 'omega' not in add_synthetic_signal:
        add_synthetic_signal['omega'] = 45
    if 'density_distribution' not in add_synthetic_signal:
        add_synthetic_signal['density_distribution'] = {'name':'2PowerLaws'}
    if 'phase_function' not in add_synthetic_signal:
        add_synthetic_signal['phase_function'] = {'name': 'HG', 'g': 0., 'polar': False}
    if 'disk_intensity_thresh' not in add_synthetic_signal:
        add_synthetic_signal['disk_intensity_thresh'] = 60
    t,n,_ = empty_data.shape
    kernel = np.fft.fft2(np.fft.fftshift(psf))
    
    # 
    # Disk injection
    # 
    my_disk = vip.metrics.scattered_light_disk.ScatteredLightDisk(nx=n,ny=n,xdo=add_synthetic_signal['xdo'], ydo=add_synthetic_signal['ydo'])
    my_disk.set_inclination(add_synthetic_signal['disk_inclination'])
    my_disk.set_pa(add_synthetic_signal['pa'])
    my_disk.set_omega(add_synthetic_signal['omega'])
    my_disk.set_flux_max(100)
    my_disk.set_density_distribution(add_synthetic_signal['density_distribution'])
    my_disk.set_phase_function(add_synthetic_signal['phase_function'])
    disk = my_disk.compute_scattered_light()

    disk = disk*(disk>add_synthetic_signal['disk_intensity_thresh'])

    disk = disk/np.max(disk)*disk_max_intensity

    # 
    # Planet injection
    # 
    if planets_positions_intensities:
        planet = np.zeros((n,n))
        for xx,yy,intensity in planets_positions_intensities:
            planet[xx,yy] = intensity
        disk += planet
    cube_disk = np.zeros((t,n,n))

    center: torch.tensor = torch.ones(1, 2)
    center[..., 0] = n / 2  # x
    center[..., 1] = n / 2  # yd
    scale: torch.tensor = torch.ones(1)

    torch_disk = torch.tensor([[disk]],requires_grad=False)

    for k in range(t):
        angle: torch.tensor = torch.ones(1) * (angles[k])
        M: torch.tensor = kornia.get_rotation_matrix2d(center, angle, scale)
        rotated_disk = kornia.warp_affine(torch_disk.float(), M, dsize=(n,n))
        cube_disk[k,:,:] = mayo_hci.A(rotated_disk[0,0,:,:].numpy(),kernel)

    data = empty_data + cube_disk
    return data, disk
