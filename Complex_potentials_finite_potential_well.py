import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Define parameters
L = 20   # Length of the box
V_wall = 10  # Potential at x=0 and x=L (finite potential walls)
sigma = 1.0  # Width of the Gaussian wave packet
dt = 0.01  # Time step
T = 1000  # Total time
h_bar = 1  # Planck's constant (set to 1 for simplicity)
Nx = 1000  # Number of spatial points
dx = L / (Nx - 1)  # Spatial step size
lambda_val =400.0  # Imaginary part of the potential at the wall

# Define the potential function with complex values at the walls
def potential(x, V_wall, L, lambda_val):
    V = np.where((x == 0), V_wall + 1j * lambda_val, 0)
    V += np.where((x == L), V_wall - 1j * lambda_val, 0)
    return V

# Create the x values
x = np.linspace(0, L, Nx)

# Initialize the wave packet with momentum
x_mid = L / 2
k0 = 52 * np.pi / L  # Wavenumber (momentum)
psi_wave_packet = np.exp(-1j * k0 * x) * np.exp(-0.5 * ((x - x_mid) / sigma)**2) / (sigma * np.sqrt(2 * np.pi))

# Normalize the wave packet
psi_wave_packet /= np.sqrt(np.trapz(np.abs(psi_wave_packet)**2, x))

# Initialize the figure and axes
fig, ax1 = plt.subplots(figsize=(30, 10))

ax1.set_xlabel('Position (x)')
ax1.set_ylabel('Potential Energy / Probability Density')
ax1.set_title('Time Evolution of Gaussian Wave Packet')

# Create a secondary y-axis for the potential
ax2 = ax1.twinx()

# Plot the potential on the secondary y-axis
ax2.plot(x, potential(x, V_wall, L, lambda_val).real, color='blue', label='Potential (V(x))')
ax2.set_ylabel('Potential Energy (V(x))')

# Initialize the line for probability density plot
line, = ax1.plot(x, np.abs(psi_wave_packet)**2, color='red', label='Probability Density')

# Define the Crank-Nicolson update function
def crank_nicolson_update(psi, V, dt, dx, h_bar):
    Nx = len(psi)
    # Construct the tridiagonal matrix for the kinetic energy operator
    alpha = 1j * h_bar * dt / (4 * dx**2)
    A = np.diag(np.ones(Nx) * (1 + 2 * alpha)) + np.diag(np.ones(Nx - 1) * (-alpha), 1) + np.diag(np.ones(Nx - 1) * (-alpha), -1)
    # Construct the right-hand side of the equation
    b = (np.diag(np.ones(Nx) * (1 - 2 * alpha)) + np.diag(np.ones(Nx - 1) * alpha, 1) + np.diag(np.ones(Nx - 1) * alpha, -1)).dot(psi)
    b += -1j * h_bar * dt * V * psi
    # Solve the system of equations using matrix inversion
    psi_new = np.linalg.solve(A, b)
    return psi_new

# Define the time evolution function
def update(frame):
    global psi_wave_packet
    # Calculate the new wave packet after a time step dt using the Crank-Nicolson scheme
    psi_wave_packet = crank_nicolson_update(psi_wave_packet, potential(x, V_wall, L, lambda_val), dt, dx, h_bar)
    # Update the probability density
    probability_density = np.abs(psi_wave_packet)**2
    # Update the plot
    line.set_ydata(probability_density)
    return line,

# Create the animation
ani = FuncAnimation(fig, update, frames=int(T/dt), blit=True, interval=30)

plt.show()