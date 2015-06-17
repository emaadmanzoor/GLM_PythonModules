"""
This module contains some auxiliary functions for processing the signals.

"""
import numpy as np
import scipy as sp
from sklearn.decomposition import PCA

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
    hwid = H.shape[1] # check thoses

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
        
    # print(np.where(spinds[j]>t0)[0])
    # print(j)
        
    # find a shortcut for find and test better
    # before I eliminate next statement
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
    
    
