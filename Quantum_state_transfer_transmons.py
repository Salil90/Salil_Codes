import matplotlib.pyplot as plt
import numpy as np
from qutip import *

wc = 8.0 * 2 * np.pi  # cavity mode frequency in GHz
wa1 = 6.0 * 2 * np.pi  # Transmon-1 frequency in GHz
wa2 = 4.0 * 2 * np.pi  # Transmon-2 frequency in GHz
g1 = 150.0 * 2 * np.pi / 1000  # coupling strength between transmon and cavity 1 in GHz (converted to GHz from MHz)
g2 = 180.0 * 2 * np.pi / 1000  # coupling strength between transmon and cavity 2 in GHz (converted to GHz from MHz)
kappa = 2.0*np.pi/100000  # cavity dissipation rate
gamma = 1.0*np.pi/10000  # Transmon dissipation rate
N = 30  # number of cavity fock states
n_th_a = 0.005  # avg number of thermal bath excitation
use_rwa = False  # Rotating wave approximation

tlist = np.linspace(0, 30, 200)

# initial state
psi0 = tensor(fock(N, 0), (basis(2, 1) + basis(2, 0)) / np.sqrt(2), basis(2, 0))  # start with an excited atom

# operators
a = tensor(destroy(N), qeye(2), qeye(2))
sm1 = tensor(qeye(N), destroy(2), qeye(2))
sm2 = tensor(qeye(N), qeye(2), destroy(2))

# Hamiltonian
if use_rwa:
    H = wc * a.dag() * a + wa1 * sm1.dag() * sm1 + g1 * (a.dag() * sm1 + a * sm1.dag()) + wa2 * sm2.dag() * sm2 + g2 * (
                a.dag() * sm2 + a * sm2.dag())
else:
    H = wc * a.dag() * a + wa2 * sm1.dag() * sm1 + g1 * (a.dag() + a) * (sm1 + sm1.dag()) + wa2 * sm2.dag() * sm2 + g2 * (
                a.dag() + a) * (sm2 + sm2.dag())

c_ops = []

# cavity relaxation
rate = kappa * (1 + n_th_a)
if rate > 0.0:
    c_ops.append(np.sqrt(rate) * a)

# cavity excitation, if temperature > 0
rate = kappa * n_th_a
if rate > 0.0:
    c_ops.append(np.sqrt(rate) * a.dag())

# qubit relaxation
rate = gamma
if rate > 0.0:
    c_ops.append(np.sqrt(rate) * sm1)

rate = gamma
if rate > 0.0:
    c_ops.append(np.sqrt(rate) * sm2)

# Projector operators for Transmon 1
ground1 = tensor(qeye(N), basis(2, 0) * basis(2, 0).dag(), qeye(2))
excited1 = tensor(qeye(N), basis(2, 1) * basis(2, 1).dag(), qeye(2))

# Projector operators for Transmon 2
ground2 = tensor(qeye(N), qeye(2), basis(2, 0) * basis(2, 0).dag())
excited2 = tensor(qeye(N), qeye(2), basis(2, 1) * basis(2, 1).dag())

# Expectation values
n_b1_ground = expect(ground1, mesolve(H, psi0, tlist, c_ops).states)
n_b1_excited = expect(excited1, mesolve(H, psi0, tlist, c_ops).states)
n_b2_ground = expect(ground2, mesolve(H, psi0, tlist, c_ops).states)
n_b2_excited = expect(excited2, mesolve(H, psi0, tlist, c_ops).states)

# Plotting
fig, ax = plt.subplots(figsize=(10, 6))

ax.plot(tlist, n_b1_ground, label="Transmon 1 Ground State", linewidth=2)
ax.plot(tlist, n_b1_excited, label="Transmon 1 Excited State", linewidth=2)
ax.plot(tlist, n_b2_ground, label="Transmon 2 Ground State", linestyle='--', linewidth=2)
ax.plot(tlist, n_b2_excited, label="Transmon 2 Excited State", linestyle='--', linewidth=2)

ax.legend(loc='upper left', bbox_to_anchor=(1, 1), fontsize=20)
ax.set_xlabel('Time (ns)', fontsize=20)
ax.set_ylabel('Population Probability', fontsize=20)
ax.set_title('Transmon Population', fontsize=20)

plt.tight_layout()
plt.show()
