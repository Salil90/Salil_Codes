import tkinter as tk
from tkinter import ttk
from scipy.integrate import solve_ivp
from scipy.integrate import odeint
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.backends.backend_tkagg as tkagg
from tkinter import Label, Entry, Button, StringVar
import math
from sympy import sin, cos, pi


class SimulatorGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Simulation Selector")

        # Create a combobox to select the simulation
        self.simulation_selection = ttk.Combobox(root, values=["Lorenz Attractor", "Chaotic Pendulum",
                                                               "Charged Particle Simulation", "Double-Slit", "Driven double pendulum"])
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

        self.double_slit_default_params = {
            "slit width": 0.01,
            "wavelength": 10.0,
            "slit distance": 2.0,
            "screen distance": 2.0,
            "screen_width": 20.0
        }

        self.driven_double_pendulum_default_params = {
            "Length of pendulum 1": 1.0,
            "Length of pendulum 2": 1.0,
            "Mass of pendulum 1": 1.0, 
            "Mass of pendulum 2": 1.0,
            "Driving frequency": 1.0,
            "Driving force": 1.0,
            "Initial angle theta 1":np.pi/2,
            "Initial angle theta 2": np.pi/4,
            "Initial angular velicuty pendulum 1": 0.0,
            "Initial angular velocity pendulum 2": 0.0,
            "Max time length  ": 100.0, 
            "Time step": 0.001, 
            "Gravitational acceleration": 9.8

        }

    def start_simulation(self):
        selected_simulation = self.simulation_selection.get()
        if selected_simulation == "Lorenz Attractor":
            self.run_lorentz_simulation()
        elif selected_simulation == "Chaotic Pendulum":
            self.run_pendulum_simulation()
        elif selected_simulation == "Charged Particle Simulation":
            self.run_charged_particle_simulation()
        elif selected_simulation == "Double-Slit":
            self.run_double_slit_simulation()
        elif selected_simulation == "Driven double pendulum":
            self.run_driven_double_pendulum_simulation()
        
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
        
        # Driven double pendulum 

    def run_driven_double_pendulum_simulation(self):
        double_pendulum_gui = tk. Toplevel()
        double_pendulum_gui.title("Chaotic dynamics in a driven double pendulum")
        

        for i, (param, default_value) in enumerate(self.driven_double_pendulum_default_params.items()):
           ttk.Label(double_pendulum_gui, text=param).grid(row=i, column=0, padx=5, pady=5)
           entry = ttk.Entry(double_pendulum_gui)
           entry.grid(row=i, column=1, padx=5, pady=5)
           entry.insert(0, default_value)
        
        def start_run_driven_double_pendulum_simulation():
            #params = [float(entry.get()) for entry in double_pendulum_gui.winfo_children() if isinstance(entry, tk.Entry)]
            params = [eval(entry.get(), {'pi': math.pi}) if '/' in entry.get() else float(entry.get()) for entry in double_pendulum_gui.winfo_children() if isinstance(entry, tk.Entry)]
            l1, l2, m1, m2, omega_d, F, theta1_initial, theta1_dot_initial, theta2_initial, theta2_dot_initial, t_max, dt, g = params
            t_span = (0, t_max)
            t_eval = np.arange(0, t_max, dt)
            state = [theta1_initial, theta1_dot_initial, theta2_initial, theta2_dot_initial]

            def double_pendulum(state, t, F, omega_d, l1, l2, m1, m2, g):
                theta1, theta1_dot, theta2, theta2_dot = state
                theta1_ddot = (-g * (2 * m1 + m2) * np.sin(theta1) - m2 * g * np.sin(theta1 - 2 * theta2) - 2 * np.sin(
                theta1 - theta2) * m2 * (theta2_dot ** 2 * l2 + theta1_dot ** 2 * l1 * np.cos(theta1 - theta2)) + F * np.sin(
                omega_d * t + theta1)) / (
                          l1 * (2 * m1 + m2 - m2 * np.cos(2 * theta1 - 2 * theta2)))
                theta2_ddot = (2 * np.sin(theta1 - theta2) * (
                theta1_dot ** 2 * l1 * (m1 + m2) + g * (m1 + m2) * np.cos(theta1) + theta2_dot ** 2 * l2 * m2 * np.cos(
                theta1 - theta2))) / (
                          l2 * (2 * m1 + m2 - m2 * np.cos(2 * theta1 - 2 * theta2)))

                return [theta1_dot, theta1_ddot, theta2_dot, theta2_ddot]

            params = [float(entry.get()) for entry in double_pendulum_gui.winfo_children() if isinstance(entry, tk.Entry)]
            l1, l2, m1, m2, omega_d, F, theta1_initial, theta1_dot_initial, theta2_initial, theta2_dot_initial, t_max, dt, g = params
            t_eval = np.arange(0, t_max, dt)
            state = [theta1_initial, theta1_dot_initial, theta2_initial, theta2_dot_initial]

            initial_state = np.array(state)
    
            # Use odeint for integration
            solution = odeint(double_pendulum, initial_state, t_eval, args=(F, omega_d, l1, l2, m1, m2, g))
            
            
            fig,axes = plt.subplots(1, 2, figsize=(12, 5))
            fig.suptitle('Double Pendulum Simulation', fontsize=16)
            axes[0].set_title('Pendulum 1')
            axes[0].set_xlabel('Theta1')
            axes[0].set_ylabel('Angular Velocity')
            axes[0].grid(True)
            line1, = axes[0].plot([], [], lw=2)

            axes[1].set_title('Pendulum 2')
            axes[1].set_xlabel('Theta2')
            axes[1].set_ylabel('Angular Velocity')
            axes[1].grid(True)
            line2, = axes[1].plot([], [], lw=2)

            def update(frame):
                if frame < len(t_eval):
                    theta1_values = solution[:frame,0]
                    omega1_values = solution[:frame,1]
                    theta2_values = solution[:frame,2]
                    omega2_values = solution[:frame,3]

                    if theta1_values.size > 0 and omega1_values.size > 0:
                        min_theta1 = min(theta1_values)
                        max_theta1 = max(theta1_values)
                        min_omega1 = min(omega1_values)
                        max_omega1 = max(omega1_values)

                        axes[0].set_xlim(min_theta1 - 0.1, max_theta1 + 0.1)
                        axes[0].set_ylim(min_omega1 - 0.1, max_omega1 + 0.1)
                        line1.set_data(theta1_values, omega1_values)


                    if theta2_values.size > 0 and omega2_values.size > 0:
                        min_theta2 = min(theta2_values)
                        max_theta2 = max(theta2_values)
                        min_omega2 = min(omega2_values)
                        max_omega2 = max(omega2_values)

                        
                        axes[1].set_xlim(min_theta2 - 0.1, max_theta2 + 0.1)
                        axes[1].set_ylim(min_omega2 - 0.1, max_omega2 + 0.1)
                        line2.set_data(theta2_values, omega2_values)
                return line1, line2


                        
            anim = FuncAnimation(fig, update, frames=len(t_eval), interval=10, blit=False)
            plt.show()

        simulate_button = ttk.Button(double_pendulum_gui, text="Simulate", command=start_run_driven_double_pendulum_simulation)
        simulate_button.grid(row=i+1, column=0, columnspan=2, pady=10)  # Adjust the row and column values as needed
            

            
    
    


        
        # Double-slit code
    def run_double_slit_simulation(self):
        double_slit_gui = tk.Toplevel()
        double_slit_gui.title("Double-Slit Parameters")

        for i, (param, default_value) in enumerate(self.double_slit_default_params.items()):
           ttk.Label(double_slit_gui, text=param).grid(row=i, column=0, padx=5, pady=5)
           entry = ttk.Entry(double_slit_gui)
           entry.grid(row=i, column=1, padx=5, pady=5)
           entry.insert(0, default_value)

        def intensity_at_point(x, x_point, slit_width, wavelength, slit_distance, screen_distance):
            intensity = 0
            for slit_offset in [-slit_distance / 2, slit_distance / 2]:
                path_difference = np.sqrt((x - slit_offset) ** 2 + screen_distance ** 2) - np.sqrt(
                (x_point - slit_offset) ** 2 + screen_distance ** 2)
                intensity += np.cos((2 * np.pi * path_difference) / wavelength) ** 2
            return intensity
        
        def start_double_slit_simulation():
            params = [float(entry.get()) for entry in double_slit_gui.winfo_children() if isinstance(entry, ttk.Entry)]
            slit_width, wavelength, slit_distance, screen_width, screen_distance = params
            fig, ax = plt.subplots()
            canvas = FigureCanvasTkAgg(fig, master=double_slit_gui)
            canvas.get_tk_widget().grid(row=len(self.double_slit_default_params), columnspan=2, padx=5, pady=5)

            def update_plot(frame):
               ax.clear()
               x = np.linspace(-screen_width / 2, screen_width / 2, 1000)
               ax.plot(x, intensity_at_point(x, -frame, slit_width, wavelength, slit_distance, screen_distance),
                    color='blue')

               slit1_x = -slit_distance / 2
               slit2_x = slit_distance / 2
               slit_height = 1.5
               ax.plot([slit1_x, slit1_x], [0, slit_height], color='red', linestyle='--', linewidth=2, label='Slits')
               ax.plot([slit2_x, slit2_x], [0, slit_height], color='red', linestyle='--', linewidth=2)

               ax.set_ylim(0, 2)
               ax.set_xlabel('Screen Position')
               ax.set_ylabel('Intensity')
               ax.set_title("Young's Double Slit Experiment")
               ax.legend()
               canvas.draw()

            frames = int(screen_distance * 10)
            interval = 100
            ani = FuncAnimation(fig, update_plot, frames=frames, interval=interval)

            def _quit():
              double_slit_gui.quit()
              double_slit_gui.destroy()
    
            #ttk.Button(double_slit_gui, text="Quit", command=_quit).grid(row=len(self.double_slit_default_params) + 1, columnspan=2, padx=5, pady=5)

            # Start the Tkinter main loop
            double_slit_gui.mainloop()

            #Update the button command to use the returned animation object
            ttk.Button(double_slit_gui, text="Start Simulation", command=lambda: ani.start()).grid(
            row=len(self.double_slit_default_params), columnspan=2, padx=5, pady=5)
            

        ttk.Button(double_slit_gui, text="Start Simulation", command=start_double_slit_simulation).grid(
        row=len(self.double_slit_default_params), columnspan=2, padx=5, pady=5)

        
        

        
    
        # Mption of charged particle
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
    plt.show()






