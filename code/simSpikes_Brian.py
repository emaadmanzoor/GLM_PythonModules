from brian import *
Vt = -50 * mvolt          # spike threshold
Vr = -60 * mvolt          # reset value
El = -49 * mvolt
tau = 20 * msecond
v0 = 0 * mV
N = 10
I_e = 0 * mV
eqs = '''dV/dt = -(V-El)/tau : volt
        I : volt
    '''
    
from scipy.linalg import block_diag
    

G = NeuronGroup(N,model = eqs, threshold = Vt,reset=Vr)
G.I = I_e

# What matrix I should pick to see some effect - check from textbook

W = np.ones((N,N))
W = np.eye(N)
W = block_diag(5*np.ones((5,5)),np.ones((5,5)))
W = block_diag(5*np.eye(5),np.eye(5))

synapse = Connection(G,G,weight = W)
#synapse = Connection(G,G,sparseness = 0.1,weight = 0.5*mvolt)
#G.v = rand(N)*10*mV

# adding randomness in the initial membrane potential
# (needs to be the same variable as the one in the equation)

#G.V = Vr + 5*rand(N) * (Vt - Vr)
G.V = v0
S = SpikeMonitor(G)
V = StateMonitor(G,'V',record = True)
net = Network(G,synapse,S,V)
net.run(1*second)
S.spiketimes
figure(1)
raster_plot(S)
figure(2)
plot(V.times /ms,V[2]/mV)

show()
net.reinit()