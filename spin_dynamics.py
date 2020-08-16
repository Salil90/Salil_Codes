
from qutip import *
import numpy as np
import matplotlib.pyplot as plt
ax = plt.gca()

# code to simulate the quantum dynamics of linear geometry with dipolar and hyperfine interactions
# part1 we consider linear geomtery with dipolar interaction between electron spins and hyperfine between one electron spin and nuclear spin
# We examine dynamics when two are same and when two are different

D11=-1
D22=-1
D33=2

A11=-1
A22=-1
A33=2
Bx=0.001
By=0.001
Bz=0.1

# Dipolar exchange part of the Hamiltonian

H11=D11*tensor(0.5*sigmax(), 0.5*sigmax(), qeye(2))
H22=D22*tensor(0.5*sigmay(), 0.5*sigmay(), qeye(2))
H33=D33*tensor(0.5*sigmaz(), 0.5*sigmaz(), qeye(2))
HD=H11+H22+H33
#print(HD)

Hf11=A11*tensor(qeye(2),0.5*sigmax(), 0.5*sigmax())
Hf22=A22*tensor(qeye(2),0.5*sigmay(), 0.5*sigmay())
Hf33=A33*tensor(qeye(2),0.5*sigmaz(), 0.5*sigmaz())
Hf=Hf11+Hf22+Hf33
HI=HD+Hf
#print(HI)  # This is the same Hamiltonian as Daniel has in his notes for linear geometry without D13

# Zeeman part of the Hamiltonian
Hz1=Bx*tensor(0.5*sigmax(), qeye(2), qeye(2))
Hz2=By*tensor(0.5*sigmay(), qeye(2), qeye(2))
Hz3=Bz*tensor(0.5*sigmaz(), qeye(2), qeye(2))
Hz4=Bx*tensor(qeye(2), 0.5*sigmax(), qeye(2))
Hz5=By*tensor(qeye(2), 0.5*sigmay(), qeye(2))
#Hz6=Bz*tensor(qeye(2), 0.5*sigmaz(), qeye(2))


Hz=Hz1+Hz2+Hz4+Hz5
H=Hz+HI

P1= tensor(0.5*sigmax(), 0.5*sigmax(), qeye(2))
P2=tensor(0.5*sigmay(), 0.5*sigmay(), qeye(2))
P3=tensor(0.5*sigmaz(), 0.5*sigmaz(), qeye(2))
Qs=0.25*tensor(qeye(2), qeye(2), qeye(2))-P1-P2-P3 # singlet projection operator
# spin permutation operators to check symmetries 
S12=tensor(sigmax(), sigmax(), qeye(2))+tensor(sigmay(), sigmay(), qeye(2))+tensor(sigmaz(), sigmaz(), qeye(2))
P12=0.5*S12+0.5*tensor(qeye(2), qeye(2), qeye(2))
Sym1=H*P12-P12*H
print(Sym1)


print(Qs)

# Initial density matrix
rhoN=0.5*tensor(qeye(2))
rhoe=singlet_state()*singlet_state().dag()
rho0=tensor(rhoe, rhoN)

K=10
c=tensor(sigmax(), qeye(2), qeye(2))

times = np.linspace(0.0, 100.0, 1000.0)
result = mesolve(H, rho0, times, [c], [Qs])
ax.plot(times, result.expect[0]);
ax.set_xlabel('Time');
ax.set_ylabel('yield');
plt.show()
plt.hold(True)




