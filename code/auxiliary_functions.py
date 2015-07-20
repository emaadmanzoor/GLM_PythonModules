"""
This module contains some auxiliary functions for processing the signals.

"""
import numpy as np
import scipy as sp
from sklearn.decomposition import PCA
import random

# Convolution filters:
def sameconv(a,b):
    """ filtered_a = sameconv(a,b)
        sameconv causally filters a with b
        
        Parameters
        ----------        

        a : array
        b : array  (1D or 2D)    
            If b is a two-dimensional array, it filters a with each column of b

        Returns
        -------
        
        filtered_a : array
            same shape as a
        
        Notes
        -----
        
        Check if there is built-in function. Check if I should use rfft.

        Check if np.convolve('same') can be used: length is max(M, N). 
    """
    if a.ndim == 1 :
        a = np.reshape(a,(a.shape[0],1))
    if b.ndim == 1:
        b = np.reshape(b,(b.shape[0],1))

    na = a.shape[0]
    nb = b.shape[0]
    n = na + nb - 1
    # a_filtered = sp.ifft(np.sum(np.tile(sp.fft(a,n,0),(1,b.shape[1]))*sp.fft(np.flipud(b),n,0),1))
    a_filtered = sp.ifft(np.tile(sp.fft(a,n,0),(1,b.shape[1]))*sp.fft(np.flipud(b),n,0),axis = 0)    
    # return(np.real(a_filtered[:na]))
    return(np.real(a_filtered[:na,:]))
    
    
def spikeconv(tsp,H,domain):
    """
        This function performs convolution of the spike train with the filter H
        
        Parameters
        ----------
        
        tsp : 1D array
            the spike_train
        H : 2D array
            the filter basis
        domain : array of length 2
            
            
                    
        Returns
        -------
    
        hcurrent : array shape (domain[1]-domain[0]+1, H.shape[1])
            the convolved spike train
    
"""

# domain should correspond to np.array([1,Stim.shape[0]/dt]))

    spinds = tsp

    hlen = H.shape[0]
    hwid = H.shape[1] 

    nsp = len(tsp)
    twin = domain

    t0 = twin[0] - hlen + 1
    t1 = twin[0] - 1
    t2 = twin[1]
    rlen = t2-t1
    hcurrent = np.zeros((rlen,hwid))

    j = 0
    while  (j<nsp) and (spinds[j]<t0):
        j = j + 1
    
    #TODO: rewrite with where and test

    j = np.arange(nsp)[np.hstack(spinds>t0)][0]
    
    if j<nsp:
        while (j<nsp) and (spinds[j]<(t2+1)):
            isp = np.round(spinds[j]) - 1
                
            i1 = isp #starting index
            if i1 < t1:
                i1 = t1
                
            imx = isp + H.shape[0]
            if imx > t2:
                imx = t2
                    

          # loop over columns
            for k in np.arange(H.shape[1]):
                for i in np.arange(i1,imx):
                    hcurrent[i-t1,k] = hcurrent[i-t1,k] + H[i-isp,k]
                    
            j = j + 1
    return(hcurrent)
    
    
    
# create a basis based on stimulus
def extractPCABasis(*args):
    """
        B = extractPCABasis(X,n)\n            
        extractPCAbasis extracts a basis of fixed dimension for data X\n
        if n is not provided, the dimension of the basis extracted automatically
        
        #TODO
        test function
        
        Note: not currently used
    """
    
    X,n = args
    # applying the probabilistic version of PCA
    if  len(args)>1:
        # PCA - n principal components
        pca = PCA(n_components = n)
    else:
        # PCA with automatic detection of number of components
        pca = PCA(n_components = 'mle')
        
    B = pca.fit_transform(X.T).T
    return(B)
            
def STA(X,y,n):
    """STA computes the spike triggered average for given spike train and stimulus. 

       Parameters
       ----------
           X : stimulus (d x T array)
           y : spike count (1 x T array)
           n : the window size for the average

       Returns
       -------
           sta : the spike-triggered average (d x n)
       
       Notes:
           here I assume I deal with numpy arrays
           I can modify to deal with data frames and time indeces
           Incorporate whitening.
           remove for loop
           process different inputs,outputs
           include covariance output
           
           #TODO
           test function
           
           Note: not currently used
    """
    
    d,T = X.shape
    idx = np.arange(T) # T is number of bins
    spike_idx = idx[y>0]
    if len(spike_idx) == 0:
        raise ValueError('Output variable y does not contain any spikes.')
            
    
    # average start from n+1 bin to have a complete history set 
    spike_idx = spike_idx[spike_idx>=n]
    if len(spike_idx) == 0:
        raise ValueError('No complete history available for spikes.')
    sta = 0     
    for i in spike_idx:
        sta = sta + X[i-n:i,]*y[i]
    sta = sta/len(spike_idx)    

    return(sta)
    
    
def simSpikes(coef,M_k,M_h,dt):
    """
        simSpikes- function to simulate
        
    """
    print(np.exp(np.dot(M,coef))[1:10])
    M = np.hstack(M_k,M_h)
    length = M.shape[0]
    print(length)
    tsp = []
    jbin = 0
    tspnext = random.expovariate(1)
    rprev = 0
    nsp = 0
    #bhlen = M_h.shape[] #TODO: change to the shape of h current 
    # ihthi = [dt:dt:max(glmprs.iht)]';  % time points for sampling
    # ihhi = interp1(glmprs.iht, ih, ihthi, 'linear', 0);
    # hlen = length(ihhi);
    chunk_size = 20 # decide on a size later
    dc = -0.7

    while jbin < length:
        indx_chunk = np.arange(jbin,min(jbin+chunk_size-1,length))

        intensity_chunk = np.exp(np.dot(M,coef)+dc)[indx_chunk]
        
        print('max indx_chunk')
        print(intensity_chunk)
        cum_intensity = np.cumsum(intensity_chunk)+rprev
        print(cum_intensity)
        print(len(cum_intensity))    
        if (tspnext>cum_intensity[-1]): 
            # No spike
            print('no spike')
            jbin = indx_chunk[-1]+1
            rprev = cum_intensity[-1]
        else: # Spike!
            print(tspnext)
            print(cum_intensity[-1])
            print('spike'+str(jbin))
            ispk = indx_chunk[np.where(cum_intensity>=tspnext)[0][0]]
            tsp.append(ispk*dt)
            postspike_limit = min(length,ispk+hlen)
            indx_postSpk = np.arange(ispk+1,postspike_limit) 
            #TODO: add if statement to record postspike current
            #TODO: pass an H current separately
            tspnext = random.expovariate(1)
            rprev = 0
            jbin = ispk + 1
            nsp = nsp + 1 
            # number of samples per iteration?
            # print(tsp[-1])
            chunk_size = max(20,np.round(1.5*jbin/nsp))#TODO: make safe for Python 2.7
    return(tsp)
