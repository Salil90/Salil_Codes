import os 
import time 
import numpy as np 
import matplotlib.pyplot as plt
import qutip as qu
import scipy
from scipy import *
from qutip import *

# Simulating annealing with effective three-body interactions between three flux qubits mediated by two coupling elements
# The effective Hamiltonian of the above model is obtained by the author of this code in their previous work: https://arxiv.org/pdf/1909.02091.pdf

N = 3  # number of spins
M = 2  # number of eigenenergies to plot

# increase taumax to get make the annealing sweep
taumax = 1000000
taulist = np.linspace(0, taumax, 100000)

# Precalculate operators
si = qu.qeye(2)
sx = qu.sigmax()
sy = qu.sigmay()
sz = qu.sigmaz()

sx_list = []
sy_list = []
sz_list = []

for n in range(N):
    op_list = []
    for m in range(N):
        op_list.append(si)

    op_list[n] = sx
    sx_list.append(qu.tensor(op_list))

    op_list[n] = sy
    sy_list.append(qu.tensor(op_list))

    op_list[n] = sz
    sz_list.append(qu.tensor(op_list))

# Define the Hamiltonian parameters 
Delta = 1.22  # Same value is used for all three qubits 
h = 0.5
J_eff = 0.09  # Effective two-body coupling 

# Construct the initial state
psi_list = [(qu.basis(2,0) + qu.basis(2,1)).unit() for n in range(N)]
psi0 = qu.tensor(psi_list)
H0 = 0 
for n in range(N):
    H0 += -0.5 * Delta * sx_list[n]

# Interaction term 
H1 = 0 
for n in range(N):
    W_factor = 0.1  # Change the weight factor to reduce or increase the strength of the effective three-body term 
    #H1 += W_factor * J_eff * sz_list[n-2] * sz_list[n-1] * sz_list[n]  # zzz interaction term 
    H1 += -J_eff * sz_list[n-2] * sz_list[n-1] - J_eff * sz_list[n-1] * sz_list[n]  # zz term

args = {'t_max': max(taulist)}
h_t = [[H0, lambda t, args: (args['t_max'] - t) / args['t_max']],
       [H1, lambda t, args: t / args['t_max']]]

# Evolve the system in time
evals_mat = np.zeros((len(taulist), M))
P_mat = np.zeros((len(taulist), M))

idx = [0]

def process_rho(tau, psi):
    # Manually evaluate the Hamiltonian at time tau
    H = h_t[0][0] * h_t[0][1](tau, args) + h_t[1][0] * h_t[1][1](tau, args)

    # Find the M lowest eigenvalues of the system
    evals, ekets = H.eigenstates(eigvals=M)

    evals_mat[idx[0], :] = np.real(evals)
    
    # Find the overlap between the eigenstates and psi 
    for n, eket in enumerate(ekets):
        P_mat[idx[0], n] = abs(eket.overlap(psi))**2
    
    idx[0] += 1

# Evolve the system
qu.mesolve(h_t, psi0, taulist, [], process_rho, args)

# Visualize the results
fig, axes = plt.subplots(2, 1, figsize=(6, 6))

# Plot eigenvalues
for n in range(len(evals_mat[0, :])):
    ls, lw = ('b', 1) if n == 0 else ('k', 0.25)
    axes[0].plot(taulist / max(taulist), evals_mat[:, n] / (2 * np.pi), ls, lw=lw)

for idx in range(len(taulist) - 1):
    for n in range(len(P_mat[0, :])):
        lw = 0.5 + 4 * P_mat[idx, n]    
        if lw > 0.55:
            axes[0].plot(np.array([taulist[idx], taulist[idx + 1]]) / taumax, 
                         np.array([evals_mat[idx, n], evals_mat[idx + 1, n]]) / (2 * np.pi), 
                         'r', linewidth=lw)

axes[0].set_xlabel(r'$s$', fontsize=15)
axes[0].set_ylabel('Eigenenergies', fontsize=15)

# Plot the occupation probabilities
for n in range(len(P_mat[0, :])):
    if n == 0:
        axes[1].plot(taulist / max(taulist), 0 + P_mat[:, n], 'r', linewidth=2)
    else:
        axes[1].plot(taulist / max(taulist), 0 + P_mat[:, n])

axes[1].set_xlabel(r'$s$', fontsize=15)
axes[1].set_ylabel('Occupation probability', fontsize=20)
axes[1].legend(("Ground state",))

plt.show()




