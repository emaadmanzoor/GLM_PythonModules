# -*- coding: utf-8 -*-
"""
    Integrate-and-fire simulation
    
    Functions:    
    simulate_LIF - standard LIF for a group of neurons
                 - they will behave the same since there is no stochastic input
    simulate_LIF_synaptic_conductance
                 - addition of synaptic input
                 - filtered spike trains of other neurons weighted by a weight matrix W
    
    
"""
import numpy as np
import time
from numpy import linalg

def simulate_LIF(E_l=-35,V0=0,V_th=-50,V_r=-60,dt=0.1,T = 10,tau = 20,I_ext=0):
    
    """ 
        This function simulates the dynamics for LIF with a threshold V_th and
        resting potential V_r with initial condition V0.
        
        For now just one neuron or many independent neurons.
        
        Returns:
        
        time_index
        V - action potential
        Y - spike trains
        
    """
    # setting some constants
    R = 1 # assuming the resistence is 1 so I do not have constant in front of I_ext
    
    # generate the time index over which the ODE will be solved
    time_index = np.arange(0,T,dt)    
    nofSteps = len(time_index)
    
    # Initializing the potential variable
    N = len(V0) # number of neurons
    V = np.zeros((N,nofSteps))
    V[:,0] = V0 # initial condition
    
    # Initializing the spike train variable
    # keep spike times or spike trains?
    # if I have the time index, I can easily subset it by Y==1     
    Y = np.zeros((N,nofSteps))
    
    # broadcasting the external current if one-dimensional
    I_ext = I_ext*np.ones((N,nofSteps))
    
    
    # Solving the ODE
    for i in range(nofSteps-1):
        
        # potential update 
        V[:,i+1] = V[:,i] - dt*((V[:,i]-E_l) + R*I_ext[:,i])/tau
        
        # reset value if there is a spike
        # (I am resetting the value that was over the threshold)
        spike = V[:,i+1]>V_th
        V[:,i+1] = (1-spike)*V[:,i+1] + spike*V_r   
        
        # creating the spike train
        Y[:,i] = spike.astype(int)
    
    return(time_index,V,Y)
    
    
def simulate_LIF_synaptic_conductance(E_l=-35,E_e=-55,V0=0,V_th=-50,V_r=-60,dt=0.1,T = 10,tau = 20,I_ext=0,W = None,Kernel = None):
    
    """ 
        This function simulates the dynamics for LIF with a threshold V_th and
        resting potential V_r with initial condition V0.
        
        For now just one neuron or many independent neurons.
        
        
        Returns:
        
        time_index
        V - action potential
        Y - spike trains
        
    """
    # setting some constants
    R = 1 # assuming the resistence is 1 so I do not have constant in front of I_ext
    
    #----------------- Variable Initialization ---------------------
    
    # generate the time index over which the ODE will be solved
    time_index = np.arange(0,T,dt)    
    nofSteps = len(time_index)
    
    # Initializing the potential variable
    N = len(V0) # number of neurons
    V = np.zeros((N,nofSteps))
    V[:,0] = V0 # initial condition
  
    # Initializing the spike train variable
    # keep spike times or spike trains?
    # if I have the time index, I can easily subset it by Y==1     
    Y = np.zeros((N,nofSteps))
    
    # broadcasting the external current if one-dimensional
    I_ext = I_ext*np.ones((N,nofSteps))  
    
    
    
    
    # pass a kernel function
    # make it to work on an array of operations
    # as long as exp it works with exponential function
    
    if Kernel is None:
        def Kernel(t,tau):
            # exponential kernel
            return(np.exp(-t/tau)/tau)
    
    
    if W is None:
        # the weighting matrix
        W = np.ones((N,N))
    
    # setting the diagonal zero so that neurons do not use their own history:        
    np.fill_diagonal(W,0) 
    
    # initializing the capacitance to zero
    g = np.zeros(V0.shape)
    
    # Solving the ODE
    for i in range(nofSteps-1):
             
        # potential update 
        V[:,i+1] = V[:,i] - dt*((V[:,i]-E_l) + (V[:,i]-E_e)*g+ R*I_ext[:,i])/tau
        
        # reset value if there is a spike
        # (I am resetting the value that was over the threshold)
        spike = V[:,i+1]>V_th
        V[:,i+1] = (1-spike)*V[:,i+1] + spike*V_r
        
        # creating the spike train
        Y[:,i] = spike.astype(int)
        
        # spike times up to time i 
        # I should maybe just accumulate the results at each iteration        
        spike_times = [time_index[idx.astype(bool)] for idx in list(Y[:,:i])]
        # some of them might be empty if there have not been spikes yet

        spike_times_filtered = ([sum(train) for train in k])

        
        
        # applying the weights:
        # I should call this synaptic conductance
        g = np.dot(W,np.array(spike_times_filtered))
        
    
    
    return(time_index,V,Y)


def simulate_synaptic_conductance(E_l=-35,E_e=-55,V0=0,V_th=-50,V_r=-60,dt=0.1,T = 10,tau = 20,I_ext=0,A_syn = None, A_gap = None):
    
    """ 
        This function simulates the dynamics for LIF with a threshold V_th and
        resting potential V_r with initial condition V0.
        
        For now just one neuron or many independent neurons.
        
        Parameters:
        
        A_syn - the adjacency matrix for the synaptic connections
        A_gap - the adjacency matrix for the gap connections
        
        
        Returns:
        
        time_index
        V - action potential
        Y - spike trains
        
    """
    
    # Auxiliary functions
    def findEquilibrium(V_leak,E,A_g,A_s,g_gap,g_syn,G_c):
        
        # --------------------------------------------------------------------------
        # To obtain equilibrium need to solve  B V_eq = c
        
        # determine a value R_m (it is the resistence)
        # R_m_i*g_hat_ij = g_gap_over_C/G_c_over_C
    
        # V_leak - is a constant?
    
        c = V_leak + (g_syn/G_c)*np.dot(A_syn,E)/2 
        B = - (g_gap/G_c)*A_gap  
        B_diag = 1 + (g_gap/G_c)*np.sum(A_gap,axis = 1) + (g_syn/G_c)*np.sum(A_syn, axis = 1)/2
        np.fill_diagonal(B,B_diag)
        V_eq = linalg.solve(B,c)
        return(V_eq)
        
    # More constants
    g_syn = 10.0 * 1e-12 # synaptic conductance (S)
    g_gap = 5.0 * 1e-12 # gap junctional conductance (S)
    G_c = 100*1e-12
        
        
        
    
    # setting some constants
    R = 1 # assuming the resistence is 1 so I do not have constant in front of I_ext
    
    #----------------- Variable Initialization ---------------------
    
    # generate the time index over which the ODE will be solved
    time_index = np.arange(0,T,dt)    
    nofSteps = len(time_index)
    
    # Initializing the potential variable
    N = len(V0) # number of neurons
    V = np.zeros((N,nofSteps))
    V[:,0] = V0 # initial condition
  
    # Initializing the spike train variable
    # keep spike times or spike trains?
    # if I have the time index, I can easily subset it by Y==1     
    Y = np.zeros((N,nofSteps))
    
    # broadcasting the external current if one-dimensional
    I_ext = I_ext*np.ones((N,nofSteps))  
    
    
    # setting the diagonal zero so that neurons do not use their own history:        
    np.fill_diagonal(W,0) 
    
    # initializing the capacitance to zero
    g = np.zeros(V0.shape)
    
    # initialize the synaptic activity variable
    s = np.zeros(V.shape)
    
    # calculate equilibrium
        
    s = findEquilibrium(V_e,E,A_gap,A_syn,g_gap,g_syn,G_c)
    
    # Solving the ODE
    for i in range(nofSteps-1):
        
        
        # Synaptic current
        I_gap = np.dot(A_syn,V[:,i])*V[:,i] - np.dot(A_syn,s*E_e)  
   
        
        # G current
        I_syn = np.sum(A_gap)*V[:,i] - np.dot(A_gap,V[:,i])
             
        # potential update 
        V[:,i+1] = V[:,i] - dt*((V[:,i]-E_l) + I_gap + I_syn + R*I_ext[:,i])/tau
        
        
    return(time_index,V)

    
# -------------------------------------------------------------------------------- 
    
#TODO Decide on Neuron and Dynamics Class Structure

    # create a group with weight matrix, indicator of which are excitory and which are inhibitory
    # specify all the constants and allow to change them
    # split what is associated with the neurons and what is associated with the dynamics
    
class NeuronGroup:    
    """ 
    A class containing multiple neuorons.
    
    Attributes:
    
    N - the number of neurons.
    W - the connectivity matrix of the neurons. 
    idx_e - idx of excitory neurons
    idx_i - idx of inhibitory neurons
    #TODO - generalize this to groups of neurons(in Brian it is a subgroup)
    # depending on the number of subgroups - number of reverse potentials
    
    
    Methods:
    
    What methods should be included: all the models?
    There are a lot of parameters in the model
    I want those to be in classes.
    
    
    """
    
    
# Should neuron group inherit stuff from individual neurons
    def __init__(self,N,W = None,subgroups = None):
        # if a weighting matrix is not provided, all weights are set to zero
        if self.W is None:
            self.W = np.zeros((N,N))
        if subgroups:
            # extract the number of subgroups
            M = len(subgroups)
            
            
        
            
    # think how to supply the kernel function and the extra parameters associated with it
    # with an exponential kernel there are no extra parameters
            
            
# A_g, A_s should be something belonging to the population of neurons
# inhibitory

# this should be a method within the population class?
# what does a group of neurons have in common?
# each individual neuron will have potential but I want to keep those in an array
# or list?
# for now make it a numpy array, maybe move it to a pandas data frame to store the times too
# in pandas data frame I can also indicate the name of the neuron
# it is harder to make it very distributed though?
# I still depend on a common matrix though

class LIF_Dynamics:
# I can store all the parameters in it
    def __init__(self,C=1,G_c=10,E_l=-35):
        self.C = C
        self.G_c = G_C
        self.E_l = E_l #mV (resting potential?)
        
    g = 100 #p
    beta = 0.125 #mV^{-1}
    a_r = 1
    a_d = 5
    
 
# Extra stuff from R code and Kunert dynamics model   
    
        # current update
#       I_g = rowSums(A_g) * V[,i] - A_g %*% V[,i]
#       I_s = A_s%*%s * V[,i]  - A_s %*% (s*E)
  
        # synaptic activity update
  #      s = s + dt*(a_r*phi(V[,i],V_eq,beta)*(1-s) - a_d*s)
  
        # synaptic inputs
        # assume 
  
        # need to specify the kernel
        # allow to pass symbolic kernel function
        # need to keep record of the spikes of the other neurons
        # extract the spike times to evaluatethe kernel
  
    
            

        
        

