""" 
    This module contains functions which generate cosine bump basis vectors 
    for the stimulus and post-spike filters.
    
    Function list:
    createRaisedCosineBasis
    createStimulusBasis
    createPostSpikeBasis
    
"""

import numpy as np
import scipy as sp


def createRaisedCosineBasis(n,peaks,b,dt):
    """ 
        Generates a basis of raised cosine bump functions.
        
        Parameters
        ---------- 
        
        n : int
            the number of basis functions
        peaks : 1D array of length 2 
            the range within which the peaks occur 
        b : float
            offset
        dt : float
            discretization step of the time domain
        

        Returns
        -------
        
        basis : array of shape 
            basis of cosine bump functions
        t_domain : 
            the time domain
            
        Notes
        -----
              
    """

    # define nonlinear functions    
    def nlin(x): return(np.log(x + 1e-20))
    def invnl(x): return(np.exp(x) - 1e-20)
  
    # generate basis of raised cosines     
    y_range = nlin(peaks+b)
    db =  np.diff(y_range)[0]/(n-1) 
    centers = np.arange(y_range[0],y_range[1]+1e-12,db)
    mxt = invnl(y_range[1]+2*db) - b
    t_domain = np.arange(0,mxt,dt)
     
    def applyCosNonlinearity(x,c,dc):
        return((np.cos(np.maximum(-sp.pi,np.minimum(sp.pi,(x-c)*sp.pi/dc/2)))+1)/2)
        
    basis = applyCosNonlinearity(np.tile(nlin(t_domain+b),[n,1]).T,np.tile(centers,[len(t_domain),1]),db)

    return(basis,t_domain)


def createStimulusBasis(pars, nkt = None):
    """
    
    Generates a basis for a stimulus filter.

    Parameters
    ----------    
    
    pars : dictionary of basis parameters
    pars['neye'] : number of identity basis vectors in front
    pars['n'] : number of vectors which are raised cosines
    pars['kpeaks'] : peak position for first and last vector
    pars['b'] : offset for linear scaling
    pars['nkt'] : number of time samples in basis (optional?)
    
    Returns
    -------
    
    kbasis : basis
    kbasis_orth : normalized basis
    t_domain : time domain

    
    Notes
    -----   
    """
    
    # Extract parameters
    neye,n,kpeaks,b = [pars['neye'],pars['n'],pars['kpeaks'],pars['b']]
   
     
    # Generate basis of raised cosines  
  
    # spacing of x axis must be of units of 1!
    kdt = 1
    kbasis0= createRaisedCosineBasis(n,kpeaks,b,kdt)[0]
    nkt0 = kbasis0.shape[0]

    # concatenate identity vectors
    a = np.hstack((np.eye(neye), np.zeros((neye,n))))
    b = np.hstack((np.zeros((nkt0,neye)), kbasis0))
    kbasis = np.vstack((a,b))
    kbasis = np.flipud(kbasis)


    # adjusting the basis time domain to nkt observations
    nkt0 = kbasis.shape[0]
    if nkt != None:
        if nkt0 < nkt: 
            kbasis = np.vstack((np.zeros((nkt-nkt0,n+neye)), kbasis))
        elif nkt0 > nkt:
            kbasis = kbasis[kbasis.shape[0]-nkt:kbasis.shape[0],:]
          
    # normalize columns
    # (normalization can return array of smaller dimension if basis is not of full rank)
    kbasis_orth = - sp.linalg.orth(kbasis)
    
    # 
    t_domain = np.arange(kbasis.shape[0])

    return(kbasis,kbasis_orth,t_domain)      
    
def createPostSpikeBasis(pars,dt):
    """
        Generates a basis for a post-spike filter.
    
        Parameters:
        ----------
        
        pars : a dictionary of basis parameters
        pars['n'] : 
            number of vectors which are raised cosines
        pars['hpeaks'] : 
            peak position for first and last vector
        pars['b'] : 
            offset for linear scaling
        pars['absref'] : 
            absolute refractory period (optional)
            
   
        
        dt : discretization step of the time domain
        
        Returns:
        --------
        hbasis : basis
        hbasis_orth : normalized basis
        t_domain : the time domain
        
    
    """
    
    # Set the parameters
    n,hpeaks,b = [pars['n'],pars['hpeaks'],pars['b']]
    if 'absref' in pars.keys():        
        absref = pars['absref']
    else:
        absref = 0
        
    if absref>dt:
        n = n - 1
    
    
    hbasis,t_domain = createRaisedCosineBasis(n,hpeaks,b,dt)
    
    
    # set first cosine basis vector bins (before 1st peak) to 1
    idx = (t_domain<=hpeaks[0])       
    hbasis[idx,0] = 1

    # create first basis vector as a step function for absolute refractory period
     
    if absref >= dt:
        ih0 = np.zeros((hbasis.shape[0],1))
        ih0[t_domain<absref,:] = 1
        hbasis[t_domain<absref,:] = 0
        hbasis = np.hstack((ih0,hbasis))
    
    # make orthogonal (the negative sign is needed to synchronize with Matlab)
    # normalization can return array of smaller dimension if basis is not of full rank    
    hbasis_orth = - sp.linalg.orth(hbasis)
    
    return(hbasis,hbasis_orth,t_domain)
    
    
