import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.fft import fft, fftfreq

# Define Pauli matrices
sigma_x = np.array([[0, 1], [1, 0]])
sigma_z = np.array([[1, 0], [0, -1]])

# Parameters
omega_0 = 5e9  # Base frequency of the defect (5 GHz)
Omega_R = 0.1e9  # Rabi frequency (100 MHz, field strength)
n_max = 100       # Number of frequency components in the comb
Delta_omega = 10e11  # Comb spacing (10 MHz)
t_span = (0, 1e-6)  # Time span for the simulation (1 microsecond)
dt = 1e-9  # Time step (1 ns)
t_eval = np.arange(t_span[0], t_span[1], dt)

# Static electric field parameters
E_field = 100000.0  # Static electric field strength (V/m)
dipole_moment = 4.0  # Dipole moment of the TLS (in C·m, or alternatively, GHz·m)

# Frequency comb function
def frequency_comb(t, omega_0, Delta_omega, n_max):
    return np.sum([np.cos((omega_0 + n * Delta_omega) * t) for n in range(-n_max, n_max + 1)], axis=0)

# Generate random frequency shifts
def generate_random_shifts(t_eval, max_shift, num_shifts):
    shift_times = np.sort(np.random.choice(t_eval, num_shifts, replace=False))
    shift_magnitudes = np.random.uniform(-max_shift, max_shift, num_shifts)
    return shift_times, shift_magnitudes

# Update omega_0 based on random shifts
def apply_random_shift(t, shift_times, shift_magnitudes, omega_0):
    shift_applied = 0
    for i, shift_time in enumerate(shift_times):
        if t >= shift_time:
            shift_applied = shift_magnitudes[i]
        else:
            break
    return omega_0 + shift_applied

# Define the Hamiltonian with random frequency shifts and static electric field
def H(t, Omega_R, omega_0, Delta_omega, n_max, E_field, dipole_moment, shift_times=None, shift_magnitudes=None):
    if shift_times is not None and shift_magnitudes is not None:
        omega_shifted = apply_random_shift(t, shift_times, shift_magnitudes, omega_0)
    else:
        omega_shifted = omega_0  # No shift case
    comb_field = Omega_R * frequency_comb(t, omega_shifted, Delta_omega, n_max)
    
    # Add the interaction with the static electric field in the σ_z term
    electric_field_interaction = dipole_moment * E_field
    
    return (omega_shifted / 2 + electric_field_interaction) * sigma_z + comb_field * sigma_x

# Define the time-dependent Schrödinger equation
def schrodinger_eq(t, psi, Omega_R, omega_0, Delta_omega, n_max, E_field, dipole_moment, shift_times=None, shift_magnitudes=None):
    psi = psi.reshape((2, 1))  # Convert to column vector
    H_t = H(t, Omega_R, omega_0, Delta_omega, n_max, E_field, dipole_moment, shift_times, shift_magnitudes)  # Time-dependent Hamiltonian
    dpsi_dt = -1j * np.dot(H_t, psi)  # Schrödinger equation
    return dpsi_dt.flatten()  # Flatten back to 1D for solver

# Renormalize the state vector to preserve norm
def renormalize(psi):
    return psi / np.linalg.norm(psi)

# Initial state (ground state)
psi_0 = np.array([1, 0], dtype=complex)

# Solve the time evolution with renormalization (optionally with random shifts)
def solve_with_shifts(t_eval, psi_0, Omega_R, omega_0, Delta_omega, n_max, E_field, dipole_moment, shift_times=None, shift_magnitudes=None):
    psi_t = np.zeros((len(t_eval), 2), dtype=complex)
    psi_t[0] = psi_0
    for i, t in enumerate(t_eval[:-1]):
        sol = solve_ivp(schrodinger_eq, (t, t_eval[i+1]), psi_t[i], args=(Omega_R, omega_0, Delta_omega, n_max, E_field, dipole_moment, shift_times, shift_magnitudes), rtol=1e-9, atol=1e-9)
        psi_t[i+1] = renormalize(sol.y[:, -1])
    return psi_t

# Compute the expectation value of sigma_x
def expectation_value_sigma_x(psi_t, sigma_x):
    return np.array([np.real(np.conj(psi) @ sigma_x @ psi) for psi in psi_t])

# Generate random shifts in the transition frequency
num_shifts = 4  # Number of random shifts
max_shift = 1e6  # Maximum shift magnitude (1 MHz)
shift_times, shift_magnitudes = generate_random_shifts(t_eval, max_shift, num_shifts)

# Plot the random shifts over time
plt.figure(figsize=(8, 6))
plt.step(shift_times, shift_magnitudes, where='post', label='Random frequency shifts', color='orange')
plt.xlabel('Time (s)')
plt.ylabel('Frequency shift (Hz)')
plt.title('Random Frequency Shifts vs. Time')
plt.grid(True)
plt.legend()
plt.tight_layout()

# Initialize a new figure for absorption spectra
plt.figure(figsize=(10, 8))

# Plot the absorption spectra in a 2x2 grid
for idx, (shift_time, shift_magnitude) in enumerate(zip(shift_times, shift_magnitudes)):
    # Perform the simulation with a specific shift
    psi_t_shift = solve_with_shifts(t_eval, psi_0, Omega_R, omega_0, Delta_omega, n_max, E_field, dipole_moment, [shift_time], [shift_magnitude])
    
    # Compute the expectation value of sigma_x over time for this shift
    expectation_sigma_x_t_shift = expectation_value_sigma_x(psi_t_shift, sigma_x)
    
    # FFT of the expectation value of sigma_x
    N = len(t_eval)
    T = dt  # Time step (1 ns)
    fft_sigma_x = fft(expectation_sigma_x_t_shift)
    fft_freqs = fftfreq(N, T)[:N//2]  # Only take the positive frequencies
    
    # Create subplots in a 2x2 grid
    plt.subplot(2, 2, idx + 1)  # 2x2 grid, place the plot in the next available spot
    plt.plot(fft_freqs / 1e9, 2.0/N * np.abs(fft_sigma_x[:N//2]), label=f'Shift at {shift_time:.2e} s with magnitude {shift_magnitude:.2e} Hz')
    plt.xlabel('Frequency (GHz)')
    plt.ylabel('Absorption Intensity')
    plt.title(f'Absorption Spectrum for Shift {idx + 1}')
    plt.grid(True)
    plt.legend()

plt.tight_layout()
plt.show()
