#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec 21 17:19:39 2019

@author: alex
"""
import numpy as np
import constants as const
from numpy import linalg as la
import math
from matplotlib import cm, colors
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.colors
import matplotlib.pyplot as plt
import scipy.special as sp
import numpy.fft as ft
import scipy.interpolate as ip
import scipy.signal as sign
import scipy.optimize as opt

def make_kernel(t, r, SHC):
    Kernel = np.zeros((len(t),len(r)))
    K0 = DEER(SHCs = SHC)
    for i in range(len(r)):        
        r1 = np.array([r[i]])    
        f1 = np.array([1])
        Kernel[:,i] = K0.trace(r1, f1, t/1) 
    return Kernel
        
def solve_tikhonov(Kernel, y, alpha):    
    aI = alpha*np.eye(Kernel.shape[1], Kernel.shape[1])
    K = np.concatenate((Kernel,aI))
    #z = np.zeros(Kernel.shape[1])
    #b = np.concatenate((trace, z))
    b = np.pad(y,(0,Kernel.shape[1]),mode='constant',constant_values=(0,0))
    f, rnorm = opt.nnls(K,b)    
    return f, rnorm

def gaussian(x, c, k):    
    return np.exp(-np.power((x - c)/k, 2.))

def pump(freq, freq1):
    tp = 1/freq1/4
    freq_eff = np.sqrt(np.power(freq,2) + np.power(freq1, 2))
    scale = np.power( freq1/freq_eff, 2)
    sin2  = np.power( np.sin(freq_eff*tp*2*const.PI)  ,2)
    return scale*sin2

def obs(freq, freq1):
    tp = 1/freq1/4
    freq_eff = np.sqrt(np.power(freq,2) + np.power(freq1, 2))
    scale = np.power( freq1/freq_eff, 5)
    sin2  = np.power( np.sin(freq_eff*tp*2*const.PI), 5)
    return scale*sin2

def integrate_sphere(f, surf):
    #print (np.sqrt( np.power(surf[:,0], 2) + np.power(surf[:,1], 2) ))
    sint = np.sqrt( np.power(surf[:,0], 2) + np.power(surf[:,1], 2) )
    #print ("now sint")
    #print (sint)
    return f@sint

def integrate_SHC_product(coeff1, coeff2):
    c1 = coeff1[0,:,:]
    c2 = coeff2[0,:,:]
    s1 = coeff1[1,:,:]
    s2 = coeff2[1,:,:]    
    return np.sum(c1*c2) - np.sum(s1*s2)
    

def plot_sphere(f, surf, nK, fmax, fmin):
    
    # PLOT THE EXCITATION PROFILE ON A SPHERE
    fcolors = f.real
    fcolors = fcolors.reshape((nK,nK))
    #fmax, fmin = fcolors.max(), fcolors.min()
    #fmax = 1
    #fmin = 0
    fcolors = (fcolors - fmin)/(fmax - fmin)
    
    # Set the aspect ratio to 1 so our sphere looks spherical
    norm = matplotlib.colors.Normalize(fmin, fmax)
    
    fig = plt.figure(figsize=plt.figaspect(1.0))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(surf[:,0].reshape((nK,nK)), surf[:,1].reshape((nK,nK)), surf[:,2].reshape(nK,nK),  rstride=1, cstride=1, facecolors=cm.jet(norm(fcolors)))
    #ax.plot_surface(surf[:,0].reshape((nK,nK)), surf[:,1].reshape((nK,nK)), surf[:,2].reshape(nK,nK),  rstride=1, cstride=1, facecolors=cm.seismic(norm(fcolors)))
    
    
    ax.view_init(30,125)    
    
    m = cm.ScalarMappable(cmap=plt.cm.jet, norm=norm)
    #m = cm.ScalarMappable(cmap=plt.cm.seismic, norm=norm)
    m.set_array([])
    #plt.colorbar(m)
    
    # Turn off the axis planes
    ax.set_axis_off()
    plt.show()
    return fig


class SpinSystem:
    """
      The definition of spin system parameters. Currently works for
      S = 1/2 and I = 1. In principle can be extended for more diverse
      nuclei and a bigger number of them.
      
      The class methods:
        get_hamiltonian(B, directional_cosines)  - calculates the Hamiltonian 
                                                   operator matrix
        eigen_freqs(B, directional_cosines)      - eigen frequencies and
                                                   eigenvectors
        transitions((B, directional_cosines))    - transition frequencies and 
                                                   their intensities
    """
    def __init__(self, **kwargs):
        self.spin = '1/2'
        self.nuclei = '14N'
        self.g_tensor = [2.002, 2.000, 2.000]
        self.hyperfine = [10, 10, 20]
         
        for key, value in kwargs.items():
            if key =='Spin':
                self.spin = value
            if key =='Nuclei':
                self.nuclei = value
            if key =='G':
                self.g_tensor = value
            if key =='A':
                self.hyperfine = value
                
        if const.N_SPIN[self.nuclei] == '1/2':
            nucl_dim = 2
        if const.N_SPIN[self.nuclei] == '1':
            nucl_dim = 3
        if self.spin == '1/2':
            elec_dim = 2
        total_dim = nucl_dim * elec_dim
        

        S_x = np.array([0, 0.5 , 0.5, 0])*(1.0+0.0j)
        S_y = np.array([0, -0.5j, 0.5j, 0])
        S_z = np.array([0.5, 0 , 0, -0.5])*(1.0+0.0j)
        S_0 = np.array([1, 0, 0, 1])        
        Sop = [S_0, S_x, S_y, S_z]
       
       
        I_z = np.array([1, 0, 0 , 0, 0, 0 , 0, 0, -1])*(1.0+0.0j)
        I_y = np.array([0, 1, 0, -1, 0, 1 , 0, -1, 0])/np.sqrt(2)/(0.0+1.0j)
        I_x = np.array([0, 1, 0 , 1, 0, 1 , 0, 1, 0 ])/np.sqrt(2)
        I_0 = np.array([ 1, 0, 0 , 0, 1, 0 , 0, 0, 1])*(1.0+0.0j)        
        Iop = [I_0, I_x, I_y, I_z]
        
        SIop = np.zeros((4, 4, len(S_0)*len(I_0)))*(1.0+0.0j)   

        self.lenS = len(S_0)
        self.lenI = len(I_0)        
        for ix, S in enumerate(Sop):            
            for jx, I in enumerate(Iop):                                
                temp = np.matmul(S.reshape(self.lenS,1),I.reshape(1,self.lenI))
                SI = temp.reshape(1, self.lenS*self.lenI)
                SI = SI.squeeze()
                SIop[ix][jx] = SI        
        self.SIop = np.zeros((4, 4, total_dim, total_dim))*(1.0+0.0j)        
        for ix in range(4):
            for jx in range(4):
                temp = SIop[ix][jx].reshape(self.lenS, self.lenI)
                M = np.block([[temp[0].reshape(nucl_dim,nucl_dim),
                               temp[1].reshape(nucl_dim,nucl_dim)],
                              [temp[2].reshape(nucl_dim,nucl_dim), 
                               temp[3].reshape(nucl_dim,nucl_dim)]])                
                self.SIop[ix][jx] = M      
    
    def __get_matrix(self, A):
        # this would work only for S=1/2 system
        temp = A.reshape(self.lenS, self.lenI)
        n_dim = int(np.round(np.sqrt(self.lenI)))        

        M = np.block([ [temp[0].reshape(n_dim,n_dim), temp[1].reshape(n_dim,n_dim)],
                       [temp[2].reshape(n_dim,n_dim), temp[3].reshape(n_dim,n_dim)] ])
        return M
  
    def get_hamiltonian(self, B, directional_cosines):

        b_x = B*directional_cosines[0]
        b_y = B*directional_cosines[1]
        b_z = B*directional_cosines[2]
        A = self.hyperfine
        G = self.g_tensor
        
        H_ez = (b_x*G[0]*self.SIop[1][0] + 
                b_y*G[1]*self.SIop[2][0] + 
                b_z*G[2]*self.SIop[3][0])*const.GAMMA_E/const.G_E/(2*const.PI)/1e6
        H_hyp= (A[0]*self.SIop[1][1]  +
                A[1]*self.SIop[2][2] +
                A[2]*self.SIop[3][3])
        H_nz = (b_x*self.SIop[0][1] + 
                b_y*self.SIop[0][2] + 
                b_z*self.SIop[0][3])*const.GAMMA_N['14N']/(2*const.PI)/1e6
        
        H = H_ez + H_hyp + H_nz
        
        return H
    
    def eigen_freqs(self, B, directional_cosines):
        H = self.get_hamiltonian(B, directional_cosines)
        evals, evecs = la.eig(H)
        return evals, evecs
    
    def transitions(self, B, directional_cosines):
        """
        Calculates transition frequencies and transition
        probabilities for of hamiltonian for a given B and its direction           
        """            
        evals, evecs = self.eigen_freqs(B, directional_cosines)
        evecs_c = np.conj(evecs)
        prob = []
        
        """
         In order to find the transition probability, one has to find first 
         the directions of an oscillating field perpendicular to the main B_z
         In fact the direction of B_1 in the x-y plane is not important for 
         the value of the matrix element
         Therefore any vector perpedicular to B can be used for the 
         direction of the oscillating field
        """        
        v1 = directional_cosines[0:3]
        if abs(v1[1]) <1e-6:
            v2 = [0, 1, 0]
        else:
            n1 = math.sqrt(1/(1+ (v1[0]/v1[1])**2))
            n2 = -n1*v1[0]/v1[1]
            v2 = np.array([n1, n2, 0])/math.sqrt(n1**2 + n2**2)
        B1_direction = v2[0]*self.SIop[1][0]+v2[1]*self.SIop[2][0]+v2[2]*self.SIop[3][0]        
        """
         la.eig has a rather non-intuitive output where the eigenvectors
         are placed as COLUMNs in the evecs output matrix, for that reason
         iterating over all the eigenvectors requires transposing the 
         evecs matrix
        """
        for vec1 in evecs.T:
            for vec2 in evecs_c.T:
                matrix_element = vec2@(B1_direction@vec1)                
                prob.append(matrix_element*np.conj(matrix_element))
        freq = []        
        for val1 in evals:
            for val2 in evals:
                freq.append(np.abs(val1 - val2))
        """
         The normalization factor of 3/2 comes from:
             three values of I projection
             another factor of 2 - not quite clear
        """
        return np.array(freq), np.array(prob)*2/3
 

class SpinSolve:
    """
      Contains the experimental parameters for calculating the spectra
      Currently contains only one method:
         epr_spectrum() which calcalutes the first order EPR spectrum 
                        (0th order derivative)
    """
    def __init__(self, spin_system, **kwargs):
        # SET UP THE DEFAULT PARAMETERS FOR THE CALCULATION
        self.mw_freq = 94900
        self.exc_band = 10
        self.n_points = 100
        self.spin_system = spin_system
        self.field = [3.37, 3.39]
        self.grid = 'DH'
        self.n_knots = 0
        self.pulse ='gaussian'
        n_knots = 0
        
        # READ THE ARGUMENTS TO PROVIDE PARAMETERS FOR THE CALCULATION
        
        for key, value in kwargs.items():
            if key =='MWFreq':
                self.mw_freq = value                
            if key =='ExcitationBandwidth':
                self.exc_band = value                
            if key =='Pulse':
                self.pulse = value
            if key =='NPoints':
                self.n_points = value 
            if key =='MagneticField':
                self.field = value
            if key == 'Grid':
                self.grid = value
            if key == 'nKnots':
                self.n_knots = value
                n_knots = self.n_knots
                
        
        if self.grid!='DH2' and self.grid!='DH':
            print("unsupported grid, defaulting to DH grid")
            self.grid = 'DH'
        if self.grid == 'DH2':
            if self.n_knots == 0:
                print("default number of nKnots = 10")
                n_knots = 10
                self.n_knots = n_knots                
            phiv = np.linspace(0, 2*const.PI*(2*n_knots - 1)/(2*n_knots),
                               2*n_knots)
            thetav = np.linspace(0, const.PI*(n_knots - 1)/(n_knots),
                                 n_knots)
            self.grid = np.meshgrid(phiv, thetav)
            
        if self.grid == 'DH':
            if self.n_knots == 0:
                print("default number of nKnots = 10")
                n_knots = 10
                self.n_knots = n_knots
            phiv = np.linspace(0, 2*const.PI*(n_knots - 1)/(n_knots),
                               n_knots)
            thetav = np.linspace(0, const.PI*(n_knots - 1)/(n_knots),
                                 n_knots)
            self.grid = np.meshgrid(phiv, thetav)
                
    
        if isinstance(self.field, list):
            self.field_range =  np.linspace(self.field[0], self.field[1],
                                        self.n_points)
         
                         
        # INITIALIZE THE DIRECTIONAL COSINES FOR A GIVEN GRID       
        phiv = self.grid[0]
        thetav = self.grid[1]
        
        x = np.sin(thetav)*np.cos(phiv)
        y = np.sin(thetav)*np.sin(phiv)
        z = np.cos(thetav)
        w = np.sin(thetav)   # weights
        
        x = x.reshape(1, x.size)
        y = y.reshape(1, y.size)
        z = z.reshape(1, z.size)
        w = w.reshape(1, w.size)

        
        directional_cosines = np.block([[x],[y],[z],[w]])
        self.directional_cosines = directional_cosines.T
  
    def epr_spectrum(self):
        #print(self.spin_system.hyperfine)
        Spec = []
        C = math.sqrt(math.log(2))
        #print (self.directional_cosines)
        # the weights of various orientations are not taken into account yet
        for B in self.field_range:
            s = []
            for dir_cos in self.directional_cosines:
                freq, prob= self.spin_system.transitions(B, dir_cos[0:3])
                if self.pulse == 'pump':
                    t = prob*pump(freq-self.mw_freq, self.exc_band)
                    #t = prob*gaussian(freq, self.mw_freq, self.exc_band)
                elif self.pulse == 'obs':
                    t = prob*obs(freq-self.mw_freq, self.exc_band)
                else:
                    t = prob*gaussian(freq, self.mw_freq, self.exc_band/C)

                s.append(np.sum(t))
            s = np.array(s)
            Spec.append(np.sum(s*self.directional_cosines[:,3]))
        
        return self.field_range, np.array(Spec)
    
    def excite_sphere(self):
        B = self.field
        sph_excitation_profile = []
        C = math.sqrt(math.log(2))
        for dir_cos in self.directional_cosines:
            freq, prob= self.spin_system.transitions(B, dir_cos[0:3])
            #t = prob*gaussian(freq, self.mw_freq, self.exc_band)  
            if self.pulse == 'pump':
                t = prob*pump(freq-self.mw_freq, self.exc_band)
            elif self.pulse == 'obs':
                t = prob*obs(freq-self.mw_freq, self.exc_band)
            else:
                t = prob*gaussian(freq, self.mw_freq, self.exc_band/C)  

            sph_excitation_profile.append(np.sum(t))    
        #return np.array(sph_excitation_profile)/self.directional_cosines[:,3], self.directional_cosines[:,0:3]
        return np.array(sph_excitation_profile), self.directional_cosines[:,0:3]

def SH_full_alpha_filter(coeff, lmax):
    temp = np.zeros_like(coeff)    
    # averaging over ALPHA angle
    for i in range(2):
        for j in range(lmax+1):
           temp[i,j,0] = coeff[i,j,0]           
    return temp

def SH_full_gamma_filter(coeff, lmax):
    return SH_full_alpha_filter(coeff, lmax)

def SH_beta_filter(beta, beta_width, djpi2, coeff, lmax):
    Sb = beta_width
    Beta = beta

    M = np.linspace(0, lmax,lmax +1)
        
    cosmpi2 = np.cos(M*np.pi/2)
    sinmpi2 = np.sin(M*np.pi/2)
    cosmb = np.cos(M*Beta)
    sinmb = np.sin(M*Beta)
    
    A = np.ones(2*lmax+2)
    A[1:(2*lmax+2):2] = -1
    
    a1 = np.ones((lmax+1,lmax+1))
    a2 = np.ones((lmax+1,lmax+1))
    
    a1[:,1:(lmax+1):2] = -1
    a2[1:(lmax+1):2,:] = -1
        
    p = np.zeros_like(djpi2)
    q = np.zeros_like(djpi2)
    r = np.zeros_like(djpi2)
    s = np.zeros_like(djpi2)
    #print(a2)

    for j in range(lmax+1):
        #print (j)
        if (j % 2) == 0:
            sign = 1
        else:
            sign = -1
        p[j,:,:] =  a1 + sign*a2 
        q[j,:,:] =  a1 - sign*a2
        r[j,:,:] =  a2 + sign*a1
        s[j,:,:] =  a2 + sign*a1

    t1 = np.swapaxes(djpi2,0,2)
    t = np.swapaxes(t1,1,2)

    p1 = p*t
    q1 = q*t
    r1 = r*t
    s1 = s*t

    #temp = np.zeros_like(coeff)
    temp = coeff
    # rotate by pi/2 around Z 
    for i in range(lmax+1):
        temp[0,i,:] =  temp[0,i,:]*cosmpi2 + temp[1,i,:]*sinmpi2
        temp[1,i,:] = -temp[0,i,:]*sinmpi2 + temp[1,i,:]*cosmpi2
    
   #rotate by pi/2 around Y
    I = np.ones(lmax+1)
    I[1:lmax+1:2] = -1
    for i in range(lmax+1):    
        temp[0,i,:] = I*p1[i,:,:]@temp[0,i,:] # C_lm coefficient
        temp[1,i,:] = I*q1[i,:,:]@temp[1,i,:] # S_lm coefficient
    
    #rotate by BETA angle
    for i in range(lmax+1):
        temp[0,i,:] =  temp[0,i,:]*cosmb + temp[1,i,:]*sinmb
        temp[1,i,:] = -temp[0,i,:]*sinmb + temp[1,i,:]*cosmb    
    
    # averaging over BETA angle
    scale_b = gaussian(M*Sb/2, 0, 1) 
    #print (scale_a)
    for i in range(2):
        for j in range(lmax+1):
           temp[i,j,:] = temp[i,j,:]*scale_b
    
    #temp = temp/np.sum(scale_b)
    
    #rotate by -pi/2 around Y
    for i in range(lmax+1):    
        temp[0,i,:] = I*r1[i,:,:]@temp[0,i,:] # C_lm coefficient
        temp[1,i,:] = I*s1[i,:,:]@temp[1,i,:] # S_lm coefficient
     
    # rotate by -pi/2 around Z
    for i in range(lmax+1):
        temp[0,i,:] =  temp[0,i,:]*cosmpi2 - temp[1,i,:]*sinmpi2 # C_lm coefficient
        temp[1,i,:] =  temp[0,i,:]*sinmpi2 + temp[1,i,:]*cosmpi2 # S_lm coefficient
    
  
    """
    I need to add nulling of the Sin coefficients for m = 0
    """
    return temp
    
def SH_gaussian_alpha_filter(alpha, width, coeff, lmax):
    Sa = width
    M = np.linspace(0, lmax,lmax +1)
 #   cosmpi2 = np.cos(M*np.pi/2)
 #   sinmpi2 = np.sin(M*np.pi/2)
    A = np.ones(2*lmax+2)
    A[1:(2*lmax+2):2] = -1
    cosma = np.cos(M*alpha)
    sinma = np.sin(M*alpha)

    temp = coeff
    # rotate over ALPHA angle
    for i in range(lmax+1):
        temp[0,i,:] =  temp[0,i,:]*cosma + temp[1,i,:]*sinma
        temp[1,i,:] = -temp[0,i,:]*sinma + temp[1,i,:]*cosma    

    #averaging over ALPHA angle
    scale_a = gaussian(M*Sa/2, 0, 1) 
    #print (scale_a)
    for i in range(2):
        for j in range(lmax+1):
           temp[i,j,:] = temp[i,j,:]*scale_a
    #temp = temp/np.sum(scale_a) 
    return temp
    
def SHGaussianFilter(width, djpi2, coeff, lmax):
    Sa = width[0]
    Sb = width[1]
    Sg = width[2]

    M = np.linspace(0, lmax,lmax +1)
    Mi = M.astype(int)
    
    cosmpi2 = np.cos(M*np.pi/2)
    sinmpi2 = np.sin(M*np.pi/2)
    A = np.ones(2*lmax+2)
    A[1:(2*lmax+2):2] = -1
    
    a1 = np.ones((lmax+1,lmax+1))
    a2 = np.ones((lmax+1,lmax+1))
    
    a1[:,1:(lmax+1):2] = -1
    a2[1:(lmax+1):2,:] = -1
        
    p = np.zeros_like(djpi2)
    q = np.zeros_like(djpi2)
    r = np.zeros_like(djpi2)
    s = np.zeros_like(djpi2)
    #print(a2)

    for j in range(lmax+1):
        #print (j)
        if (j % 2) == 0:
            sign = 1
        else:
            sign = -1
        p[j,:,:] =  a1 + sign*a2 
        q[j,:,:] =  a1 - sign*a2
        r[j,:,:] =  a2 + sign*a1
        s[j,:,:] =  a2 + sign*a1

    t1 = np.swapaxes(djpi2,0,2)
    t = np.swapaxes(t1,1,2)

    p1 = p*t
    q1 = q*t
    r1 = r*t
    s1 = s*t

    temp = coeff[:,:(lmax+1),:(lmax+1)]
    
   # averaging over ALPHA angle
    scale_a = gaussian(M*Sa/2, 0, 1) 
    #print (scale_a)
    for i in range(2):
        for j in range(lmax+1):
           temp[i,j,:] = temp[i,j,:]*scale_a
            
    # rotate by pi/2 around Z 
    for i in range(lmax+1):
        temp[0,i,:] =  temp[0,i,:]*cosmpi2 + temp[1,i,:]*sinmpi2
        temp[1,i,:] = -temp[0,i,:]*sinmpi2 + temp[1,i,:]*cosmpi2
    
   #rotate by pi/2 around X
    I = np.ones(lmax+1)
    I[1:lmax+1:2] = -1
    for i in range(lmax+1):    
        temp[0,i,:] = I*p1[i,:,:]@temp[0,i,:] # C_lm coefficient
        temp[1,i,:] = I*q1[i,:,:]@temp[1,i,:] # S_lm coefficient
    
    # averaging over BETA angle
    
    # produce an array with Fourier coeffiecients for abs(sin beta)
    N1 = np.linspace(0,2*lmax,2*lmax+1)
    b = np.zeros(len(N1))    
    b[0] = 2/math.pi
    b[1] = 0
    b[2:(2*lmax+1)] = -2/math.pi*(1 + (-1)**N1[2:(2*lmax+1)])/(np.power(N1[2:(2*lmax+1)],2) -1)
  
    # Test that Fourier Series coefficients are correct
    #x = np.linspace(-math.pi, math.pi, 100)
    #s = np.zeros_like(x)
    #for n in range(lmax+1):
    #    s = s + b[n]*np.cos(n*x)
    
    # calculate scaling factor
    N2 = np.linspace(-(2*lmax),2*lmax+1, 4*lmax+1)
    expn = 1/2*gaussian(N2*Sb/2, 0, 1)
    scale_b = np.zeros_like(M)
    for m in Mi:
        expnp = expn[ m+2*lmax: (m+3*lmax+1)]
        expnm = expn[-m+2*lmax:(-m+3*lmax+1)]
        scale_b[m] = b[0:(lmax+1)]@(expnp+expnm)
    scale_b = scale_b/scale_b[0]
    # now apply scaling due to BETA averaging
    for i in range(2):
        for j in range(lmax+1):
            temp[i,j,:] = temp[i,j,:]*scale_b    
        
    #rotate by -pi/2 around X
    for i in range(lmax+1):    
        temp[0,i,:] = I*r1[i,:,:]@temp[0,i,:] # C_lm coefficient
        temp[1,i,:] = I*s1[i,:,:]@temp[1,i,:] # S_lm coefficient
     
    # rotate by -pi/2 around Z
    for i in range(lmax+1):
        temp[0,i,:] =  temp[0,i,:]*cosmpi2 - temp[1,i,:]*sinmpi2 # C_lm coefficient
        temp[1,i,:] =  temp[0,i,:]*sinmpi2 + temp[1,i,:]*cosmpi2 # S_lm coefficient
    
    # averaging over GAMMA angle
    scale_g = gaussian(M*Sg/2, 0, 1) 
    #print (scale_g)
    for i in range(2):
        for j in range(lmax+1):
            temp[i,j,:] = temp[i,j,:]*scale_g
    """
    I need to add nulling of the Sin coefficients for m = 0
    """
    
    return temp



def SHCylindricalFilter(beta, beta_width, djpi2, coeff, lmax):
    Sb = beta_width
    Beta = beta

    M = np.linspace(0, lmax,lmax +1)
    Mi = M.astype(int)
    
    cosmpi2 = np.cos(M*np.pi/2)
    sinmpi2 = np.sin(M*np.pi/2)
    cosmb = np.cos(M*Beta)
    sinmb = np.sin(M*Beta)
    
    A = np.ones(2*lmax+2)
    A[1:(2*lmax+2):2] = -1
    
    a1 = np.ones((lmax+1,lmax+1))
    a2 = np.ones((lmax+1,lmax+1))
    
    a1[:,1:(lmax+1):2] = -1
    a2[1:(lmax+1):2,:] = -1
        
    p = np.zeros_like(djpi2)
    q = np.zeros_like(djpi2)
    r = np.zeros_like(djpi2)
    s = np.zeros_like(djpi2)
    #print(a2)

    for j in range(lmax+1):
        #print (j)
        if (j % 2) == 0:
            sign = 1
        else:
            sign = -1
        p[j,:,:] =  a1 + sign*a2 
        q[j,:,:] =  a1 - sign*a2
        r[j,:,:] =  a2 + sign*a1
        s[j,:,:] =  a2 + sign*a1

    t1 = np.swapaxes(djpi2,0,2)
    t = np.swapaxes(t1,1,2)

    p1 = p*t
    q1 = q*t
    r1 = r*t
    s1 = s*t

    temp = np.zeros_like(coeff)    

   # averaging over ALPHA angle
    for i in range(2):
        for j in range(lmax+1):
           temp[i,j,0] = coeff[i,j,0]           
            
    # rotate by pi/2 around Z 
    for i in range(lmax+1):
        temp[0,i,:] =  temp[0,i,:]*cosmpi2 + temp[1,i,:]*sinmpi2
        temp[1,i,:] = -temp[0,i,:]*sinmpi2 + temp[1,i,:]*cosmpi2
    
   #rotate by pi/2 around X
    I = np.ones(lmax+1)
    I[1:lmax+1:2] = -1
    for i in range(lmax+1):    
        temp[0,i,:] = I*p1[i,:,:]@temp[0,i,:] # C_lm coefficient
        temp[1,i,:] = I*q1[i,:,:]@temp[1,i,:] # S_lm coefficient
    
    #rotate by BETA angle
    for i in range(lmax+1):
        temp[0,i,:] =  temp[0,i,:]*cosmb + temp[1,i,:]*sinmb
        temp[1,i,:] = -temp[0,i,:]*sinmb + temp[1,i,:]*cosmb    
    
    # averaging over BETA angle
    scale_b = gaussian(M*Sb/2, 0, 1) 
    #print (scale_a)
    for i in range(2):
        for j in range(lmax+1):
           temp[i,j,:] = temp[i,j,:]*scale_b
  
    
    """
      this is another version of averaging procedure
      where the sin(beta) factor is taken into account - 
      which is actually not needed
    # averaging over BETA angle
    
    # produce an array with Fourier coeffiecients for abs(sin beta)
    N1 = np.linspace(0,2*lmax,2*lmax+1)
    b = np.zeros(len(N1))    
    b[0] = 2/math.pi
    b[1] = 0
    b[2:(2*lmax+1)] = -2/math.pi*(1 + (-1)**N1[2:(2*lmax+1)])/(np.power(N1[2:(2*lmax+1)],2) -1)
  
    # Test that Fourier Series coefficients are correct
    #x = np.linspace(-math.pi, math.pi, 100)
    #s = np.zeros_like(x)
    #for n in range(lmax+1):
    #    s = s + b[n]*np.cos(n*x)
    
    # calculate scaling factor
    N2 = np.linspace(-(2*lmax),2*lmax+1, 4*lmax+1)
    expn = 1/2*gaussian(N2*Sb/2, 0, 1)
    scale_b = np.zeros_like(M)
    for m in Mi:
        expnp = expn[ m+2*lmax: (m+3*lmax+1)]
        expnm = expn[-m+2*lmax:(-m+3*lmax+1)]
        scale_b[m] = b[0:(lmax+1)]@(expnp+expnm)
    scale_b = scale_b/scale_b[0]
    # now apply scaling due to BETA averaging
    for i in range(2):
        for j in range(lmax+1):
            temp[i,j,:] = temp[i,j,:]*scale_b    
    """
        
    #rotate by -pi/2 around X
    for i in range(lmax+1):    
        temp[0,i,:] = I*r1[i,:,:]@temp[0,i,:] # C_lm coefficient
        temp[1,i,:] = I*s1[i,:,:]@temp[1,i,:] # S_lm coefficient
     
    # rotate by -pi/2 around Z
    for i in range(lmax+1):
        temp[0,i,:] =  temp[0,i,:]*cosmpi2 - temp[1,i,:]*sinmpi2 # C_lm coefficient
        temp[1,i,:] =  temp[0,i,:]*sinmpi2 + temp[1,i,:]*cosmpi2 # S_lm coefficient
    
    # averaging over GAMMA angle
    temp2 = np.zeros_like(temp)    
    for i in range(2):
        for j in range(lmax+1):            
            temp2[i,j,0] = temp[i,j,0]           

    """
    I need to add nulling of the Sin coefficients for m = 0
    """
    return temp2

class ModelDEER:
    def __init__(self, **kwargs):
        self.degree = 8
        self.l = 10000
        omega_max = 10
        dt = 1/(2*omega_max)
        self.omega = np.linspace(-omega_max,omega_max,self.l)
        self.time = np.linspace(0, dt*self.l,self.l)
        self.gauss = gaussian(self.omega, 0, 0.01)        
        for key, value in kwargs.items():
            if key =='Degree':
                self.degree = value
                
    def dipolar_spectrum(self, f, order):
        Spec = np.zeros_like(f)
        I1 = np.argwhere(f<=-2)
        I2 = np.argwhere(np.logical_and(f>-2, f<-1))
        I3 = np.argwhere(np.logical_and(f>=-1, f<=1))
        I4 = np.argwhere(np.logical_and(f>1, f<2))
        I5 = np.argwhere(f>=2)
        
        Spec[I1] = 0
        
        z1 = np.sqrt( (1 - f[I2])/3);
        t1 = np.arccos(np.sqrt(z1))
        Spec[I2] = 1/z1 * np.real(sp.sph_harm(0,self.degree,0,t1))
        
        z1 = np.sqrt( (1 - f[I3])/3);
        t1 = np.arccos(np.sqrt(z1))
        z2 = np.sqrt( (1 + f[I3])/3);
        t2 = np.arccos(np.sqrt(z2))
        Spec[I3] = 1/z1 * np.real(sp.sph_harm(0,self.degree,0,t1)) + \
                   1/z2 * np.real(sp.sph_harm(0,self.degree,0,t2))
                   
        z1 = np.sqrt( (1 + f[I4])/3);
        t1 = np.arccos(np.sqrt(z1))
        Spec[I4] = 1/z1 * np.real(sp.sph_harm(0,self.degree,0,t1))
        
        Spec[I5] = 0
        return Spec
    
    def trace(self):
        Spec = self.dipolar_spectrum(self.omega,self.degree)
        Spec_conv = sign.convolve(Spec, self.gauss, mode ='same')
        Spec_conv = np.roll(Spec_conv,self.l//2)
        trace = np.real(ft.ifft(Spec_conv))        
        return ip.interp1d(self.time,trace,kind = 'cubic')

class DEER:
    def __init__(self,**kwargs):
        self.SHCs = np.array([1])
        for key, value in kwargs.items():
            if key =='SHCs':
                self.SHCs = value
        self.all_traces = []
        for i in range(len(self.SHCs)):
            m = ModelDEER(Degree = i*2)
            self.all_traces.append(m.trace())
    def model(self, time):
        sum1 = 0
        for i in range(len(self.SHCs)):
            sum1 = sum1 + self.all_traces[i](time)*self.SHCs[i]
        return sum1
    
    def trace(self, r, f, time):
        freq_dd = 52.16/np.power(r,3)        
        sum1 = np.zeros_like(time)
        for j in range(len(self.SHCs)): 
            model = self.all_traces[j]
            for i in range(len(freq_dd)):
                sum1 = sum1 + f[i]*model(freq_dd[i]*time)*self.SHCs[j]  #    
        return sum1