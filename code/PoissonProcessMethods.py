"""
PoissonProcessMethods module contains functions for simulating and estimating
nonhomogenous Poisson processes.    
"""



import numpy as np
from scipy import optimize
from scipy import random
import unittest
import warnings


def sampleHomogeneousPP(l,N):
    """
        This function samples from a homogenous Poisson process. 
        
        Parameters
        ----------
        
        l : float
            
            the rate of the Poisson process
            
        N : int
            the number of observations from the process

        
        Returns
        -------
        eventTimes : array of shape (N,)
            - the time at which the events occur
        
        
        Notes
        -----
        
        There could be multiple observations occurring at the same time.
                        
    """
    interArrivalTimes = [int(random.expovariate(1./l)) for i in np.arange(N)]
    eventTimes = np.cumsum(interArrivalTimes)
    return(eventTimes)
    
    
    
    
def sampleEvents(intensity,dt):
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
    u = np.random.uniform(size = len(intensity))
    return(intensity*dt>u)




def sampleInhomogeneousPP(intensity,dt,I):
    """
        Samples time events on an interval [0,N)
        according to an inhomogeneous Poisson process with
        conditional intensity lambda
        
        Parameters
        ----------

        intensity : array
        I : time interval [0,N]           
        
        
        
                
        
    """
    Lambda = sum(intensity)*dt
    n = np.random.poisson(Lambda)
    u = np.random.uniform(0,I[1],size = n)
    #TODO need inverse
    print ('Need to calculate the inverse of the intensity')
    raise KeyboardInterrupt
    
    
def samplePiecewiseConstantPP(intensity,dt):
    """
        Samples observations from a Poisson process with a piecewise-constant
        intensity.

        Parameters
        ----------
        
        intensity : 1D array
            the intensity evaluated at each bin
        dt : float
            the length of the bin
            
        Returns
        -------
        y : 1D array (same shape as intensity) 
            the number of events in each bin
        
    """
    y = np.apply_along_axis(np.random.poisson,0,l*dt)
    return(y)
    


def poissonRegression(X,y,theta_0,dt=1,f = 'exp',method = 'L-BFGS-B'):
    """ poissonRegression computes an estimate for the coefficients theta
                        
        Parameters
        ----------
        X : a 2D array of covariates
        y : a 1D array of outputs 
        dt : the spacing between the observations
        theta_0 : initial guess 
        f : str or callable
            The inverse link function: default is exp, but one can pass any function.
            ! Some functions may break the convergence of the algorithm.
            
        
        Returns
        -------
        theta_MLE : array
        
        Example
        -------
        
        >>> theta = np.random.normal(size = (2,))
        >>> X = 0.1*np.random.normal(size = (2,1000))
        >>> l = np.exp(np.dot(X.T,theta))
        >>> Y = sampleEvents(l,0.01) 
        >>> theta_MLE = poissonRegression(X,Y,dt,np.zeros(theta.shape))
        
        

        Notes
        ------
        - provide a model structure to change
        - index of columns to use (factors)        
        - the pdf (assuming independence of bins: use transition density otherwise)
        - information about dt
        - provide statistics for the fit of the regression?        
    """
    
    
    
    def negLogL(theta,X,y,dt,f):
        """
           theta : parameters
           X : covariates
           y : response
           dt : discretization step
           f : nonlinear transformation of the input (default is exp)
           
        """
        # calculate the intensity of the Poisson process
        #intensity = np.exp(np.dot(X.T,theta))# for log link
        
        if f == 'exp': 
            intensity = np.exp(np.dot(X.T,theta))
        else:        
            intensity = f(np.dot(X.T,theta))[0]
        # bins with spikes and bins with no spikes
        l = sum(intensity)*dt - sum(y*np.log(intensity))
        
        return(l)
        
    
    def grad(theta,X,y,dt,f):
        """
           theta : parameters
           X : covariates
           y : response
           dt : discretization step
           f : nonlinear transformation of the input (default is exp)
           (should return the function value and its derivative)
           
        """
        if f == 'exp': 
            # log-link
            intensity = np.exp(np.dot(X.T,theta)) 
            g = np.dot(X,intensity)*dt - np.dot(X,y) 
        else:
            intensity,d_intensity = f(np.dot(X.T,theta))
            g = np.dot(X,d_intensity)*dt - np.dot(X,(y*intensity/d_intensity))
               
        return(g)  
        
    #TODO
    # def Hessian
            

    opts = {'disp':True,'maxiter': 200}
    theta_hat = optimize.minimize(negLogL,theta_0, jac = grad, args = (X,y,dt,f),options = opts,method = method).x
    # theta_hat = optimize.minimize(negLogL,theta_0, args = (X,y,dt),options = opts).x
    return(theta_hat)


class testPoissonProcessMethods(unittest.TestCase):
    def test_passing_nonlinearity(self):
        """ 
            This function tests the passing a user-defined inverse-link function. 
            Passing a user defined exponential function should coincide with the
            default exponential function.
        """
        theta = np.random.normal(size = (10,))
        X = 0.1*np.random.normal(size = (10,100))
        # generate spike trains according to the poisson process
        
        
    
        l = np.exp(np.dot(X.T,theta))
        dt = 0.01
        Y = sampleEvents(l,dt) 
        
        def myExponential(x):
            return(np.exp(x),np.exp(x))
        theta_0 = np.zeros((X.shape[0]))
        theta_hat1 = poissonRegression(X,Y,theta_0,0.1,myExponential)
        theta_hat2 = poissonRegression(X,Y,theta_0,0.1,'exp')
        
        #self.assertEqual(theta_hat1,theta_hat2)
        np.testing.assert_array_almost_equal(theta_hat1,theta_hat2)
    
    def test_estimation(self):
        N = 10000
        d = 2
        theta = np.random.normal(size = (d,))
        X = 0.1*np.random.normal(size = (d,N))
        l = np.exp(np.dot(X.T,theta))
        dt = 0.001
        Y = sampleEvents(l,dt) 
        theta_0 = np.zeros(theta.shape)
        theta_MLE = poissonRegression(X,Y,theta_0,dt)
        error = sum(np.abs(theta_MLE - theta))
        tol = 5*d
        # print(error)
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
        l = np.exp(np.dot(X.T,theta))
        dt = 0.1
        Y = sampleEvents(l,dt) 
        theta_0 = np.zeros(theta.shape)
        theta_MLE = poissonRegression(X,Y,theta_0,dt)
        l_predicted = np.exp(np.dot(X.T,theta_MLE))
        Y_predicted = sampleEvents(l_predicted,dt)
        total = sum(Y+Y_predicted)
        if total != 0:
            error_rate = sum(np.abs(Y - Y_predicted)).astype('float64')/total
        else:
            error_rate = 0
            warnings.warn('No events observed.')
            
        tol = .1
        # print('error_rate'+str(error_rate))
        # print(sum(np.abs(Y - Y_predicted)))
        # print(sum(Y+Y_predicted)/2.)
        self.assertTrue(error_rate < tol)  
        
    def test_simulation(self):
        """
            This function tests how good the simulation of the Poisson process is.
            
        """
        N = 1000
        intensity = 3*np.ones((N,))
        y = sampleEvents(intensity,0.1)
        tol = 0.1
        self.assertTrue(np.sum(np.abs(sum(y).astype('Float64'))/len(intensity) - 0.3)<tol,msg = 'Incomplete test')

    def test_oneInput(self): 
        raise NotImplementedError
                

if __name__ == '__main__':
    unittest.main()
    