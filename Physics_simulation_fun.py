import tkinter as tk
from tkinter import ttk
from scipy.integrate import solve_ivp
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg


class SimulatorGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Simulation Selector")

        # Create a combobox to select the simulation
        self.simulation_selection = ttk.Combobox(root, values=["Lorenz Attractor", "Chaotic Pendulum",
                                                               "Charged Particle Simulation"])
        self.simulation_selection.grid(row=0, column=0, padx=5, pady=5)

        # Button to start the selected simulation
        self.start_button = ttk.Button(root, text="Start Simulation", command=self.start_simulation)
        self.start_button.grid(row=0, column=1, padx=5, pady=5)

        # Default parameters
        self.lorentz_default_params = {
            "Sigma": 10.0,
            "Beta": 8 / 3,
            "Rho": 28.0,
            "X Initial": 1.0,
            "Y Initial": 1.0,
            "Z Initial": 1.0,
            "Time": 100.0,
            "Samples": 10000
        }

        self.pendulum_default_params = {
            "Initial Angle (radians)": 0.5,
            "Initial Angular Velocity (radians/s)": 0.0,
            "Damping Coefficient": 0.1,
            "Amplitude": 1.0,
            "Frequency": 0.5,
            "Simulation Time (s)": 30.0,
            "Time Step (s)": 0.01
        }

        self.charged_particle_default_params = {
            "Electric Field Ex": 0.0,
            "Electric Field Ey": 0.0,
            "Electric Field Ez": 1.0,
            "Magnetic Field Bx": 0.0,
            "Magnetic Field By": 1.0,
            "Magnetic Field Bz": 0.0,
            "Damping Coefficient": 0.1,
            "Time Step (dt)": 1e-12
        }

    def start_simulation(self):
        selected_simulation = self.simulation_selection.get()

        if selected_simulation == "Lorenz Attractor":
            self.run_lorentz_simulation()
        elif selected_simulation == "Chaotic Pendulum":
            self.run_pendulum_simulation()
        elif selected_simulation == "Charged Particle Simulation":
            self.run_charged_particle_simulation()

    def run_lorentz_simulation(self):
        lorentz_gui = tk.Toplevel()
        lorentz_gui.title("Lorentz Attractor Parameters")

        for i, (param, default_value) in enumerate(self.lorentz_default_params.items()):
            ttk.Label(lorentz_gui, text=param).grid(row=i, column=0, padx=5, pady=5)
            entry = ttk.Entry(lorentz_gui)
            entry.grid(row=i, column=1, padx=5, pady=5)
            entry.insert(0, default_value)

        def start_lorentz_simulation():
            params = [float(entry.get()) for entry in lorentz_gui.winfo_children() if isinstance(entry, tk.Entry)]
            sigma, beta, rho, x0, y0, z0, time, samples = params

            t_span = (0, time)
            t_eval = np.linspace(0, time, int(samples))

            xyz0 = [x0, y0, z0]

            def lorentz(t, xyz, sigma, beta, rho):
                x, y, z = xyz
                dxdt = sigma * (y - x)
                dydt = x * (rho - z) - y
                dzdt = x * y - beta * z
                return [dxdt, dydt, dzdt]

            def integrate_lorentz(sigma, beta, rho, t_span, xyz0, t_eval):
                sol = solve_ivp(lorentz, t_span, xyz0, t_eval=t_eval, args=(sigma, beta, rho))
                return sol.y

            data = integrate_lorentz(sigma, beta, rho, t_span, xyz0, t_eval)

            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
            ax.set_xlim(-20, 20)
            ax.set_ylim(-30, 30)
            ax.set_zlim(0, 50)
            ax.set_xlabel('X-axis')
            ax.set_ylabel('Y-axis')
            ax.set_zlabel('Z-axis')
            ax.set_title('Lorentz Attractor')

            line, = ax.plot([], [], [], lw=1)

            def update_plot(frame):
                line.set_data(data[0, :frame], data[1, :frame])
                line.set_3d_properties(data[2, :frame])
                return line,

            anim = FuncAnimation(fig, update_plot, frames=data.shape[1], interval=10, blit=True)
            plt.show()

        ttk.Button(lorentz_gui, text="Start Simulation", command=start_lorentz_simulation).grid(
            row=len(self.lorentz_default_params), columnspan=2, padx=5, pady=5)

    def run_pendulum_simulation(self):
        pendulum_gui = tk.Toplevel()
        pendulum_gui.title("Chaotic Pendulum Parameters")

        for i, (param, default_value) in enumerate(self.pendulum_default_params.items()):
            ttk.Label(pendulum_gui, text=param).grid(row=i, column=0, padx=5, pady=5)
            entry = ttk.Entry(pendulum_gui)
            entry.grid(row=i, column=1, padx=5, pady=5)
            entry.insert(0, default_value)

        def start_pendulum_simulation():
            params = [float(entry.get()) for entry in pendulum_gui.winfo_children() if isinstance(entry, tk.Entry)]
            theta0, omega0, damping, amplitude, frequency, simulation_time, time_step = params

            g = 9.81
            l = 1.0

            t_span = (0, simulation_time)
            t_eval = np.arange(0, simulation_time, time_step)

            def pendulum_equation(t, state, damping, amplitude, frequency):
                theta, omega = state
                dtheta_dt = omega
                domega_dt = - (g / l) * np.sin(theta) - damping * omega + amplitude * np.sin(frequency * t)
                return [dtheta_dt, domega_dt]

            solution = solve_ivp(pendulum_equation, t_span, [theta0, omega0], t_eval=t_eval,
                                 args=(damping, amplitude, frequency))

            fig, ax = plt.subplots()
            ax.set_xlabel('Angle (radians)')
            ax.set_ylabel('Angular Velocity (radians/s)')
            ax.set_title('Chaotic Pendulum Motion')

            line, = ax.plot([], [], lw=2)

            def update(frame):
                if frame < len(t_eval):
                    theta_values = solution.y[0, :frame]
                    omega_values = solution.y[1, :frame]

                    if theta_values.size > 0 and omega_values.size > 0:
                        min_theta = min(theta_values)
                        max_theta = max(theta_values)
                        min_omega = min(omega_values)
                        max_omega = max(omega_values)

                        ax.set_xlim(min_theta - 0.1, max_theta + 0.1)
                        ax.set_ylim(min_omega - 0.1, max_omega + 0.1)

                        line.set_data(theta_values, omega_values)

                return line,

            anim = FuncAnimation(fig, update, frames=len(t_eval), interval=10, blit=True)
            plt.show()

        ttk.Button(pendulum_gui, text="Start Simulation", command=start_pendulum_simulation).grid(
            row=len(self.pendulum_default_params), columnspan=2, padx=5, pady=5)

    def run_charged_particle_simulation(self):
        charged_particle_gui = tk.Toplevel()
        charged_particle_gui.title("Charged Particle Simulation Parameters")

        for i, (param, default_value) in enumerate(self.charged_particle_default_params.items()):
            ttk.Label(charged_particle_gui, text=param).grid(row=i, column=0, padx=5, pady=5)
            entry = ttk.Entry(charged_particle_gui)
            entry.grid(row=i, column=1, padx=5, pady=5)
            entry.insert(0, default_value)

        def start_charged_particle_simulation():
            params = [float(entry.get()) for entry in charged_particle_gui.winfo_children() if
                      isinstance(entry, tk.Entry)]
            Ex, Ey, Ez, Bx, By, Bz, damping, dt = params

            root = tk.Toplevel()
            app = ChargedParticleSimulation(root, Ex, Ey, Ez, Bx, By, Bz, damping, dt)
            root.mainloop()

        ttk.Button(charged_particle_gui, text="Start Simulation",
                   command=start_charged_particle_simulation).grid(row=len(self.charged_particle_default_params),
                                                                    columnspan=2, padx=5, pady=5)


class ChargedParticleSimulation:
    def __init__(self, root, Ex, Ey, Ez, Bx, By, Bz, damping_coefficient, dt):
        self.root = root
        self.root.title("Charged Particle Simulation")

        # Parameters for charged particle simulation
        self.Ex = Ex
        self.Ey = Ey
        self.Ez = Ez
        self.Bx = Bx
        self.By = By
        self.Bz = Bz
        self.damping_coefficient = damping_coefficient
        self.dt = dt

        # Initialize trajectory
        self.r = np.zeros((1, 3))
        self.v = np.array([[1.0e6, 0.0, 0.0]])

        # Create GUI elements
        self.create_gui()

    def create_gui(self):
        # Labels and Entry boxes for electric field components
        ttk.Label(self.root, text="Electric Field Ex:").grid(row=0, column=0, sticky=tk.W)
        self.entry_Ex = ttk.Entry(self.root)
        self.entry_Ex.grid(row=0, column=1)
        self.entry_Ex.insert(0, str(self.Ex))

        ttk.Label(self.root, text="Electric Field Ey:").grid(row=1, column=0, sticky=tk.W)
        self.entry_Ey = ttk.Entry(self.root)
        self.entry_Ey.grid(row=1, column=1)
        self.entry_Ey.insert(0, str(self.Ey))

        ttk.Label(self.root, text="Electric Field Ez:").grid(row=2, column=0, sticky=tk.W)
        self.entry_Ez = ttk.Entry(self.root)
        self.entry_Ez.grid(row=2, column=1)
        self.entry_Ez.insert(0, str(self.Ez))

        # Labels and Entry boxes for magnetic field components
        ttk.Label(self.root, text="Magnetic Field Bx:").grid(row=3, column=0, sticky=tk.W)
        self.entry_Bx = ttk.Entry(self.root)
        self.entry_Bx.grid(row=3, column=1)
        self.entry_Bx.insert(0, str(self.Bx))

        ttk.Label(self.root, text="Magnetic Field By:").grid(row=4, column=0, sticky=tk.W)
        self.entry_By = ttk.Entry(self.root)
        self.entry_By.grid(row=4, column=1)
        self.entry_By.insert(0, str(self.By))

        ttk.Label(self.root, text="Magnetic Field Bz:").grid(row=5, column=0, sticky=tk.W)
        self.entry_Bz = ttk.Entry(self.root)
        self.entry_Bz.grid(row=5, column=1)
        self.entry_Bz.insert(0, str(self.Bz))

        # Entry box for damping coefficient
        ttk.Label(self.root, text="Damping Coefficient:").grid(row=6, column=0, sticky=tk.W)
        self.entry_damping = ttk.Entry(self.root)
        self.entry_damping.grid(row=6, column=1)
        self.entry_damping.insert(0, str(self.damping_coefficient))

        # Entry box for time step
        ttk.Label(self.root, text="Time Step (dt):").grid(row=7, column=0, sticky=tk.W)
        self.entry_dt = ttk.Entry(self.root)
        self.entry_dt.grid(row=7, column=1)
        self.entry_dt.insert(0, str(self.dt))

        # Button to start simulation
        ttk.Button(self.root, text="Simulate", command=self.simulate).grid(row=8, column=0, columnspan=2)

        # Create canvas for Matplotlib plot
        self.fig = plt.figure(figsize=(8, 6))
        self.ax = self.fig.add_subplot(111, projection='3d')
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.root)
        self.canvas.draw()
        self.canvas.get_tk_widget().grid(row=9, column=0, columnspan=2)

    def update_parameters(self):
        # Update parameters from Entry boxes
        self.Ex = float(self.entry_Ex.get())
        self.Ey = float(self.entry_Ey.get())
        self.Ez = float(self.entry_Ez.get())
        self.Bx = float(self.entry_Bx.get())
        self.By = float(self.entry_By.get())
        self.Bz = float(self.entry_Bz.get())
        self.damping_coefficient = float(self.entry_damping.get())
        self.dt = float(self.entry_dt.get())

    def simulate(self):
        # Update parameters
        self.update_parameters()

        # Clear previous plot
        self.ax.clear()
        self.ax.set_xlabel('X (m)')
        self.ax.set_ylabel('Y (m)')
        self.ax.set_zlabel('Z (m)')
        self.ax.set_title('Charged Particle Trajectory')

        # Initialize trajectory
        self.r = np.zeros((1, 3))
        self.v = np.array([[1.0e6, 0.0, 0.0]])

        # Create animation
        self.ani = FuncAnimation(self.fig, self.update_plot, frames=1000, interval=10, blit=False)

        # Start the animation
        self.ani._start()

    def update_plot(self, frame):
        # Calculate trajectory with damping term
        q = 1.6e-19  # Charge of the particle (C)
        m = 9.11e-31  # Mass of the particle (kg)

        a = (q / m) * (np.array([self.Ex, self.Ey, self.Ez]) + np.cross(self.v[-1], np.array([self.Bx, self.By, self.Bz])))
        damping_force = - self.damping_coefficient * self.v[-1]  # Damping force proportional to velocity
        new_v = self.v[-1] + (a + damping_force) * self.dt
        new_r = self.r[-1] + new_v * self.dt
        self.v = np.vstack((self.v, new_v))
        self.r = np.vstack((self.r, new_r))

        # Plot trajectory
        self.ax.plot(self.r[:, 0], self.r[:, 1], self.r[:, 2])
        self.canvas.draw()


if __name__ == "__main__":
    root = tk.Tk()
    app = SimulatorGUI(root)
    root.mainloop()









