# -*- coding: utf-8 -*-

"""
This module contains some auxiliary functions for processing the signals.

"""
import numpy as np
import scipy as sp
from sklearn.decomposition import PCA
import random
import copy
import unittest

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
    
    
def simSpikes(coef,M_k,H=None,ht_domain=None,dt=1):
    """
        simSpikes - function to simulate a neuronal activity by sampling a Poisson process 
        with conditional intensity based on the stimulus, with an option to add the effect
        of the postspike kernel after each spike
        
        Inputs
        ------
        coef: 
            a coefficient containing the coefficients corresponding to the stimulus effects,
            constant effects and the postspike effects if there are any

        M_k: 
            covariate for the effects of the stimulus and the constant
        H: 
            post-spike filtering matrix
        ht_domain: 
            the domain of the post-spike filter
        dt: 
            time step
            
        Returns
        -------
        
        tsp: 
            a list of the spike times
            
        Notes
        -----
        Instead of directly passing M_k, I can pass the k-filter and the constant separately.
        Then I can interpolate according to dt.
        
    
    """
    
    # TODO: add check for the correct number of coefficients 
    # (depending on whether H is provided)
    
    coeff_k = coef[:M_k.shape[1]]
    cov_effects = np.dot(M_k,coeff_k)
    
    if H is not None:
        coeff_h = coef[M_k.shape[1]:]
        ih = np.dot(H,coeff_h)
        hlen = len(ih)
        # interpolate with new positions
        ih = np.interp(np.arange(dt,ht_domain[-1],dt),ht_domain,ih)


    length = M_k.shape[0]
    jbin = 0
    rprev = 0
    nsp = 0
    chunk_size = 100 # decide on a size later
    tsp = []
    tspnext = random.expovariate(1)


    while jbin < length:
        indx_chunk = np.arange(jbin,min(jbin+chunk_size-1,length)).astype(int)
        intensity_chunk = np.exp(cov_effects[indx_chunk])
        cum_intensity = np.cumsum(intensity_chunk)*dt/100+rprev
        
        print(tspnext)
        print(cum_intensity[-1])
   
        if (tspnext>cum_intensity[-1]): 
            # No spike
            print('no spike')
            jbin = indx_chunk[-1]+1
            rprev = cum_intensity[-1]
        else: # Spike!
            print('spike'+str(jbin))
            ispk = indx_chunk[np.where(cum_intensity>=tspnext)[0][0]]
            tsp.append(ispk*dt)
            nsp = nsp + 1 
            
            # postspike effect if H is provided
            if H is not None: 
                
                postspike_limit = min(length,ispk+hlen)
                indx_postSpk = np.arange(ispk+1,postspike_limit)
                
                # adding the post-spike effect
                cov_effects[indx_postSpk] = cov_effects[indx_postSpk]+ih[:(postspike_limit - ispk)-1]
                
                
            tspnext = random.expovariate(1)
            rprev = 0
            jbin = ispk + 1          
            chunk_size = max(20,np.round(1.5*jbin/nsp)) #TODO: make safe for Python 2.7
    
    return(tsp)
    
def simSpikesMultiple(coeff_list,M_k,H=None,ht_domain=None,dt=1):
    """
        simSpikes - function to simulate a neuronal activity for multiple Neurons by sampling a Poisson process 
        with conditional intensity based on the stimulus, with an option to add the postspike effects from other neurons

        Inputs
        ------
        coef: list
            a list containing the coefficients corresponding to the stimulus effects,
            constant effects and the postspike effects for each neuron

        M_k: 
            covariate for the effects of the stimulus and the constant
        
        H: 
            post-spike filtering matrix
        
        ht_domain: 
            the domain of the post-spike filter
        
        dt: 
            time step
            
        Note: 
            We assume that the Stimulus is the same and filtered with the same stimulus matrix,
            but we allow to change the coefficients. We also assume that the post-spike and coupling
            kernels are the same - not necessary. At this stage we expect the user to provide both
            post-spike and coupling effects, there is no option to provide post-spike effects only,
            but one can simply set the coefficients to zero.
            
            
        Returns
        -------
            tsp_list: a list which contains the lists of spike times for each neuron
        
    """
    
    # add check for the correct number of coefficients 
    # (depending on whether H is provided)
    
    nofNeurons = len(coeff_list)
    cov_effects = []
    ih = []
    
    for i in range(nofNeurons):
        
        coeff_k = coeff_list[i][:M_k.shape[1]]
        cov_effects.append(np.dot(M_k,coeff_k))
    
            
    # convert to an nd array
    cov_effects = np.array(cov_effects).T

    
    if H is not None:
        for i in range(nofNeurons):
            coeff_h = coeff_list[i][M_k.shape[1]:]
            #ih.append(np.dot(H,np.reshape(coeff_h,(H.shape[1],nofNeurons),order = 'F')))

            # interpolate with new positions
            # skip the interpolation for now
            ih_interpolated = np.zeros((len(np.arange(dt,ht_domain[-1],dt)),nofNeurons))
            for j in range(nofNeurons):
                ih_interpolated[:,j] = np.interp(np.arange(dt,ht_domain[-1],dt),ht_domain,np.dot(H,np.reshape(coeff_h,(H.shape[1],nofNeurons),order = 'F'))[:,j])
            ih.append(ih_interpolated)
                # hlen = len(ih)
            hlen = H.shape[0]
            

    length = M_k.shape[0]
    jbin = 0
    rprev = [0]*nofNeurons
    rprev =np.zeros((nofNeurons,))
    chunk_size = 100 # decide on a size later
    # create a list of lists to hold the spike times for each neuron
    tsp_list = [[] for i in range(nofNeurons)]
    tspnext_list = [random.expovariate(1) for i in range(nofNeurons)]
    nsp = np.zeros(nofNeurons)



    
    
    while jbin < length:
        indx_chunk = np.arange(jbin,min(jbin+chunk_size-1,length)).astype(int)
        intensity_chunk = np.exp(cov_effects[indx_chunk,:])
        cum_intensity = np.cumsum(intensity_chunk,axis=0)*dt/100+rprev
        sum(cum_intensity)
        
        print(tspnext_list)
        print(cum_intensity[-1,:])
   
        if (all(tspnext_list>cum_intensity[-1,:])): 
            # No spike
            print('no spike')
            jbin = indx_chunk[-1]+1
            rprev = cum_intensity[-1,:]
        else: # Spike!
            print('spike'+str(jbin))     
            ispks,jspks = np.where(cum_intensity>=tspnext_list)

            spikingNeuronIndx = np.unique(jspks[ispks == min(ispks)])
            ispk = indx_chunk[min(ispks)]
            rprev = cum_intensity[min(ispks),:];
            
            # add each spike
            for j in range(len(spikingNeuronIndx)):
                idx = spikingNeuronIndx[j]
                tsp_list[idx].append(ispk*dt)
                nsp[idx] = nsp[idx] + 1 
            
                # postspike effect if H is provided
                if H is not None: 
                    postspike_limit = min(length,ispk+hlen)
                    indx_postSpk = np.arange(ispk+1,postspike_limit)
                
                    # adding the post-spike effect
                    cov_effects[indx_postSpk,:] = cov_effects[indx_postSpk,:]+ih[idx][:(postspike_limit - ispk)-1,:]
                rprev[idx] = 0
                tspnext_list[idx] = random.expovariate(1)

            jbin = ispk + 1
            chunk_size = max(20,np.round(1.5*jbin/sum(nsp))) #TODO: make safe for Python 2.7
    
    return(tsp_list)
    
    
    
def makeInterpMatrix(slen,nofBins):
    """
        
    """    
    
    c1 = np.arange(1/nofBins,1+1/nofBins,1/nofBins)
    cc = np.hstack((c1,c1[::-1] - 1/nofBins))
    M = np.zeros((nofBins*slen,slen))
    for j in range(slen-1):
        M[j*nofBins:(j+2)*nofBins,j] = cc  
    M[nofBins*slen-nofBins:nofBins*slen,slen-1] = c1;
    return(M)
    
def makeInterpMatrix2(slen,nofBins):
    """
        Note: assumes slen < nofBins
    """
    c1 = np.arange(1/nofBins,1+1/nofBins,1/nofBins)
    cc = np.hstack((c1,c1[::-1] - 1/nofBins))
    M = np.zeros((nofBins*slen,slen))
    phi = np.floor(nofBins/2)

    M[:nofBins+phi,0] = cc[phi:]
    for j in np.arange(1,slen-1):
        M[j*nofBins-phi:(j+2)*nofBins-phi,j] = cc
    M[nofBins*slen-nofBins - phi:nofBins*slen,slen-1] = cc[:nofBins+phi];
    return(M)
    

    
# --------------- Unit Testing --------------------
class testAuxiliaryFunctions(unittest.TestCase):
    def testSimSpikesMultipleWithSingleNeuron(self):
        """
            This test is testing whether sampling with simSpikesMultiple and
            coefficients for single neuron provides the same spike train as
            sampling with simSpikes.
            
            Notes
            -----
            
            I might want to change the domain for interpolation.            
            
        """
        
        import filters
        
        # generate stimulus
        Stim = np.round(sp.rand(2500)*4-2);
        
        
        dt = 0.01
        
        # create a stimulus filter
        kpeaks = np.array([0,round(20/3)])
        pars_k = {'neye':5,'n':5,'kpeaks':kpeaks,'b':3}
        K,K_orth,kt_domain = filters.createStimulusBasis(pars_k, nkt = 20) 
        
        # create a post-spike filter
        hpeaks = np.array([0.1,2])
        pars_h = {'n':5,'hpeaks':hpeaks,'b':.4,'absref':0.}
        H,H_orth,ht_domain = filters.createPostSpikeBasis(pars_h,dt)
        
        # convolving the stimulus with each basis function
        Stim_convolved = sameconv(Stim,K)
        
        # interpolate on a new grid
        

        m = Stim_convolved.shape[0]
        new_domain = np.arange(0,dt*m,dt)
        M_k = np.zeros((len(new_domain),K.shape[1])) # this will fail for one dimensional K
        for i in np.arange(K.shape[1]):
            M_k[:,i] = np.interp(new_domain,np.arange(m),Stim_convolved[:,i])
        #M_k = interp(np.arange(),np.range(Stim_convolved.shape[],Stim_convolved)
        print(M_k.shape)
        M = np.hstack((M_k,np.ones((M_k.shape[0],1))))
        
        # setting up the coefficients
        coeff_k0 = np.array([ 0.061453,0.284916,0.860335,1.256983,0.910615,0.488660,-0.887091,0.097441,0.026607,-0.090147])
        coeff_h0 = np.array([-10, -5, 0, 2, -2])
        pars0 = np.hstack((coeff_k0,3,coeff_h0))
        
        # 
        random.seed(0)
        tsp_single = simSpikes(pars0,M,H,ht_domain,dt = 0.01)
        
        print(tsp_single)
        
        # 
        random.seed(0)
        tsp_multiple = simSpikesMultiple([pars0],M,H,ht_domain,dt = 0.01)
        
        # self.assertTrue(True)
        self.assertTrue(tsp_single==tsp_multiple[0])
        
    
# Test which compares the results of simSpikes and simSpikesMultiple when
# simSpikesMultiple has only one neuron
    
# need to set the fix the random stream
    
if __name__ == '__main__':
    unittest.main()
    
    
