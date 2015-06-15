"""
    This module contains classes for simulation and estimation of non-homogeneous Poisson processes.
"""

import numpy as np
from scipy import optimize
from scipy import random
import unittest
import warnings

class PPModel(object):
    """
        
        Class PPModel provides methods to sample and estimate parameters in nonhomogenous-time Poisson 
        processes. 
        
        Attributes
        ----------
        
        
        covariates : 2D array
            the array of covariates
        dt : float 
            the time discretization
        coef : 1D array 
            parameters: not needed to initialize the object
        f : str or callable
            nonlinear function of the covariates
                    
    """    
    
    
    def __init__(self,covariates,coef=None,dt=1,f = 'exp'):
        self.covariates = covariates
        self.dt = dt
        self.coef = coef
        self.f = f
        
        
    def negLogL(self,coef,y):
        """
            Calculate the negative log-likelohood.
            
            Parameters
            ----------
            
            coef : parameters
            y : response
            
            Returns
            -------
            
            l : the negative log-likelihood
            
        """
        
        # check if y has correct shape
        if y.shape!=(self.covariates.shape[1],):
            raise ValueError(str(y.shape)+str(self.covariates.shape[0])+'y should be a 1D array with length equal to dimension of the covariates.')
    
        # calculate the intensity of the Poisson process
        if self.f == 'exp': 
            intensity = np.exp(np.dot(self.covariates.T,coef)) # log-link
        else:        
            intensity = self.f(np.dot(self.covariates.T,coef))[0]
        
        # bins with events and bins with no events
        l = sum(intensity)*self.dt - sum(y*np.log(intensity))
        
        return(l)
        
    def gradNegLogL(self,coef,y):
        """
            Calculate the gradient of the negative log-likelihood.
            
            Parameters
            ----------
            
            coef : parameters
            y : response          
            
            Returns
            -------
            
            g : the gradient
        """
        if self.f == 'exp': 
            # log-link
            intensity = np.exp(np.dot(self.covariates.T,coef)) 
            g = np.dot(self.covariates,intensity)*self.dt - np.dot(self.covariates,y) 
        else:
            intensity,d_intensity = self.f(np.dot(self.covariates.T,coef))
            g = np.dot(self.covariates,d_intensity)*self.dt - np.dot(self.covariates,(y*intensity/d_intensity))
               
        return(g) 
        
        
    def hessNegLogL(self,coef,y):
        """
            Calculate the Hessian of the negative log-likelihood.
            
            Parameters
            ----------
            
            coef : parameters
            y : response   
            
            Returns
            -------
            
            H : the Hessian
        """
        if self.f == 'exp':
            intensity = np.exp(np.dot(self.covariates.T,coef)) 
            
            D = np.diag(intensity)
            #H = X^TDX
            H = np.dot(np.dot(self.covariates,D),self.covariates)
        # else:
            # finish writing derivative
        return(H)
    
    
    def fit(self, y,start_coef=None, method='L-BFGS-B', maxiter=400, disp=True):
        """  
        Computes an estimate for the unknown coefficients based on response y.
                        
        Parameters
        ----------
        y : a 1D array of outputs 
        dt : the spacing between the observations
        start_coef : initial guess (if not given set to zeros)
        method : minimization method
            Should be one of
            ‘Nelder-Mead’
            ‘Powell’
            ‘CG’
            ‘BFGS’
            ‘Newton-CG’
            ‘Anneal (deprecated as of scipy version 0.14.0)’
            ‘L-BFGS-B’
            ‘TNC’
            ‘COBYLA’
            ‘SLSQP’
            ‘dogleg’
            ‘trust-ncg’
            custom - a callable object (added in version 0.14.0)
        maxiter : int
            the maximum number of iterations
        disp : Boolean
            if True display minimization results
            
        
        Returns
        -------
        res: OptimizeResult object
        res.x : estimate for the coefficient
        res.success : Boolean
        res.fun : function value
        
        Notes
        ------
        """
        
        
        opts = {'disp':disp,'maxiter':maxiter}
        if start_coef==None:
            start_coef = np.zeros((self.covariates.shape[0],))
        res = optimize.minimize(self.negLogL,start_coef,jac = self.gradNegLogL,hess = self.hessNegLogL, args = y, options = opts, method = method)
        return(res)
        
    
    def sampleEvents(self,coef):
        """
            Generates a sequence of events based on a Poisson process with
            a time-dependent intensity.
            
            Parameters
            ----------
            intensity : an array of 
            dt: float
                the spacing between two evaluations of the intensity
            
            Returns
            -------
            an array (same length as intensity)
            = 1 - event occurs
            = 0 - event does not occur
        
        """
        if self.f=='exp':
            intensity = np.exp(np.dot(self.covariates.T,coef))
        else:
            intensity = self.f(np.dot(self.covariates.T,coef))[0]
        #TODO
        # raise error if coef empty
        u = np.random.uniform(size = len(intensity))   
        y = (intensity*self.dt>u)          
        return(y.astype(int))
        
    def samplePiecewiseConstantPP(self,coef):
        """
        Samples observations from a Poisson process with a piecewise-constant
        intensity.

        Parameters
        ----------
        
        coef: parameters
            
        Returns
        -------
        y : 1D array (same shape as intensity) 
            the number of events in each bin
        
        """
        if self.f=='exp':
            intensity = np.exp(np.dot(self.covariates.T,coef))
        else:
            intensity = self.f(np.dot(self.covariates.T,coef))[0]
            
        y = np.apply_along_axis(np.random.poisson,0,intensity*self.dt)
        return(y)
        


# --------------- Unit Testing --------------------
class testPoissonProcessClasses(unittest.TestCase):
    def test_passing_nonlinearity(self):
        """ 
            This function tests the passing a user-defined inverse-link function. 
            Passing a user defined exponential function should coincide with the
            default exponential function.
        """
        theta = np.random.normal(size = (10,))
        X = 0.1*np.random.normal(size = (10,100))
        # generate spike trains according to the Poisson process     
        
        
        def myExponential(x):
            return(np.exp(x),np.exp(x))
        theta_0 = np.zeros((X.shape[0]))
        
        model1 = PPModel(X,coef = theta,f = myExponential,dt = 0.1)
        model2 = PPModel(X,coef = theta,f = 'exp',dt = 0.1)
    
        y = model2.sampleEvents(theta)
        
        theta_hat1 = model1.fit(y,theta_0).x
        theta_hat2 = model2.fit(y,theta_0).x
        
        np.testing.assert_array_almost_equal(theta_hat1,theta_hat2)
    
    def test_estimation(self):
        """ 
            This test  tests the estimation performance when 
            N is large and d is small. 
        """
        N = 10000
        d = 2
        theta = np.random.normal(size = (d,))
        X = 0.1*np.random.normal(size = (d,N))
        theta_0 = np.zeros(theta.shape)
        model = PPModel(X,coef = theta,dt = 0.001)
        y = model.sampleEvents(theta)
        theta_MLE = model.fit(y,theta_0).x
        error = sum(np.abs(theta_MLE - theta))
        tol = 5*d
        self.assertTrue(error < tol, msg = "Estimation failed with error tolerance = 5d")   
        
    def test_prediction(self):
        """
            This function is testing the prediction when l = exp(theta).
            In this case each observation comes from a Poisson distribution with
            a rate theta_i so the estimate for theta will rely on whether there
            was an event in the bin or not, which results in a good prediction 
            (though not so good estimation)
            
        """        
        
        N = 100
        theta = np.random.normal(size = (N,))
        X = np.eye(N)
        dt = 0.1
     
        theta_0 = np.zeros(theta.shape)
        model = PPModel(X,coef = theta,dt = dt)
        Y = model.sampleEvents(theta)
        theta_MLE = model.fit(Y,theta_0).x
        Y_predicted = model.sampleEvents(theta_MLE)
        total = sum(Y+Y_predicted)
        if total != 0:
            error_rate = sum(np.abs(Y - Y_predicted)).astype('float64')/total
        else:
            error_rate = 0
            warnings.warn('No events observed.')
            
        tol = .1
        self.assertTrue(error_rate < tol)  
        
    def test_simulation(self):
        """
            This function tests how good the simulation of the Poisson process is.
            
        """
        N = 1000
        intensity = 3*np.ones((N,))
        model = PPModel(np.eye(N),dt = 0.1)
        y = model.sampleEvents(3*np.ones((N,)))
        tol = 1
        self.assertTrue(np.sum(np.abs(sum(y).astype('Float64'))/len(intensity) - 0.3)<tol,msg = 'Incomplete test')

    def test_oneInput(self): 
        #TODO
        # complete test
        return(0)        
        #raise NotImplementedError
                          

if __name__ == '__main__':
    unittest.main()
    

    
        
