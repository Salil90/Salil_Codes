import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
plt.rc('xtick',labelsize=20)
plt.rc('ytick',labelsize=20)
import numpy as np
from qutip import *
from scipy import *




N = 9   # number of spins
M = 2  # number of eigenenergies to plot

# array of spin energy splittings and coupling strengths (random values). 
h  = 1.0 * 2 * pi * (1 - 2 * rand(N))
Jz = 1.0 * 2 * pi * (1 - 2 * rand(N))
Jx = 1.0 * 2 * pi * (1 - 2 * rand(N))
Jy = 1.0 * 2 * pi * (1 - 2 * rand(N))

# increase taumax to get make the sweep more adiabatic
taumax = 10.0
taulist = np.linspace(0, taumax, 1000)
#Precalculate operators
# pre-allocate operators
si = qeye(2)
sx = sigmax()
sy = sigmay()
sz = sigmaz()

sx_list = []
sy_list = []
sz_list = []

for n in range(N):
    op_list = []
    for m in range(N):
        op_list.append(si)

    op_list[n] = sx
    sx_list.append(tensor(op_list))

    op_list[n] = sy
    sy_list.append(tensor(op_list))

    op_list[n] = sz
    sz_list.append(tensor(op_list))
#Construct the initial state
psi_list = [(basis(2,0)+basis(2,1)).unit() for n in range(N)]
psi0 = tensor(psi_list)
H0 = 0    
for n in range(N):
    H0 +=  -0.5  * sx_list[n]
#for n in range(N-1):
 #   H0 += -1.0*sx_list[n]*sx_list[n+1]
#Construct the Hamiltonian
# energy splitting terms
H1 = 0    
for n in range(N):
    H1 += - 0.5 * h[n] * sz_list[n]

H1 = 0    
for n in range(N-2):
    # interaction terms
    #H1 += - 0.5 * Jx[n] * sx_list[n] * sx_list[n+1]
    #H1 += - 0.5 * Jy[n] * sy_list[n] * sy_list[n+1]
    H1 += - 0.5 * Jz[n] * sz_list[n] * sz_list[n+1]*sz_list[n+2]

# the time-dependent hamiltonian in list-function format
args = {'t_max': max(taulist)}
h_t = [[H0, lambda t, args : (args['t_max']-t)/args['t_max']],
       [H1, lambda t, args : t/args['t_max']]]
#Evolve the system in time
#
# callback function for each time-step
#
evals_mat = np.zeros((len(taulist),M))
P_mat = np.zeros((len(taulist),M))

idx = [0]
def process_rho(tau, psi):
  
    # evaluate the Hamiltonian with gradually switched on interaction 
    H = qobj_list_evaluate(h_t, tau, args)

    # find the M lowest eigenvalues of the system
    evals, ekets = H.eigenstates(eigvals=M)

    evals_mat[idx[0],:] = real(evals)
    
    # find the overlap between the eigenstates and psi 
    for n, eket in enumerate(ekets):
        P_mat[idx[0],n] = abs((eket.dag().data * psi.data)[0,0])**2    
        
    idx[0] += 1
# Evolve the system, request the solver to call process_rho at each time step.

mesolve(h_t, psi0, taulist, [], process_rho, args)
#Odedata object with sesolve data.

#states = True, expect = True
#num_expect = 0, num_collapse = 0
#Visualize the results
#Plot the energy levels and the corresponding occupation probabilities (encoded as the width of each line in the energy-level diagram).

#rc('font', family='serif')
#rc('font', size='10')

fig, axes = plt.subplots(2, 1, figsize=(6,6))

#
# plot the energy eigenvalues
#

# first draw thin lines outlining the energy spectrum
for n in range(len(evals_mat[0,:])):
    ls,lw = ('b',1) if n == 0 else ('k', 0.25)
    axes[0].plot(taulist/max(taulist), evals_mat[:,n] / (2*pi), ls, lw=lw)

# second, draw line that encode the occupation probability of each state in 
# its linewidth. thicker line => high occupation probability.
for idx in range(len(taulist)-1):
    for n in range(len(P_mat[0,:])):
        lw = 0.5 + 4*P_mat[idx,n]    
        if lw > 0.55:
           axes[0].plot(array([taulist[idx], taulist[idx+1]])/taumax, 
                        array([evals_mat[idx,n], evals_mat[idx+1,n]])/(2*pi), 
                        'r', linewidth=lw)    
        
axes[0].set_xlabel(r'$s$', fontsize=15)

axes[0].set_ylabel('Eigenenergies', fontsize=15)

#axes[0].set_title("Energyspectrum (%d lowest values) of a chain of %d spins.\n " % (M,N)
                #+ "The occupation probabilities are encoded in the red line widths.")


#
# plot the occupation probabilities for the few lowest eigenstates
#
for n in range(len(P_mat[0,:])):
    if n == 0:
        axes[1].plot(taulist/max(taulist), 0 + P_mat[:,n], 'r', linewidth=2)
    else:
        axes[1].plot(taulist/max(taulist), 0 + P_mat[:,n])

axes[1].set_xlabel(r'$s$', fontsize=15)
axes[1].set_ylabel('Occupation probability', fontsize=15)
#axes[1].set_title("Occupation probability of the %d lowest " % M +
                 # "eigenstates for a chain of %d spins" % N)
axes[1].legend(("Ground state",));

plt.show()
