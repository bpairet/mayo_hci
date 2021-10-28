import torch

def SLsheardec2D_pytorch(X, shearlets):
#def SLsheardec2D(X, shearletSystem):
    _,_,n_shearlets = shearlets.shape
    
    coeffs = torch.zeros(shearlets.shape, dtype=torch.complex128, device= X.device)
    Xfreq = torch.fft.fftshift(torch.fft.fft2(torch.fft.ifftshift(X)))
    for j in range(n_shearlets):
        coeffs[:,:,j] = torch.fft.fftshift(torch.fft.ifft2(torch.fft.ifftshift(Xfreq*torch.conj(shearlets[:,:,j]))))
    if torch.imag(coeffs).max()>5e-8:
        print("Warning: magnitude in imaginary part exceeded 5e-08.")
        print("Data is probably not real-valued. Largest magnitude: " + str(torch.imag(coeffs).max()))
        print("Imaginary part neglected.")
    return torch.real(coeffs)


def SLshearadjoint2D_pytorch(coeffs, shearlets):    
#def SLshearadjoint2D(coeffs, shearletSystem):
    _,_,n_shearlets = shearlets.shape
    
    #X = np.zeros((coeffs.shape[0], coeffs.shape[1]), dtype=complex)
    X = torch.zeros((coeffs.shape[0], coeffs.shape[1]), dtype=torch.complex128, device= coeffs.device)

    #for j in range(shearletSystem["nShearlets"]):
    for j in range(n_shearlets):
        #X += np.conj(shearletSystem["shearlets"][:,:,j]) * fftlib.fftshift(fftlib.fft2(fftlib.ifftshift(coeffs[..., j])))
        X += torch.conj(shearlets[:,:,j]) * torch.fft.fftshift(torch.fft.fft2(torch.fft.ifftshift(coeffs[..., j])))
    #Xresult = fftlib.fftshift(fftlib.ifft2(fftlib.ifftshift(X)))
    Xresult = torch.fft.fftshift(torch.fft.ifft2(torch.fft.ifftshift(X)))
    return torch.real(Xresult)#.astype(coeffs.dtype)