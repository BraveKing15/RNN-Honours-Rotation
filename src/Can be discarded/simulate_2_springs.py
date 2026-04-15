
import numpy as np
import matplotlib.pyplot as plt

# Parameters from the paper for the first 2 masses/springs
# Masses (kg)
m = np.array([1/4, 1/3])
# Damping coefficients (Ns/m)
c = np.array([1/4, 1/3])
# Spring constants (N/m)
k = np.array([1, 5/6])

def gamma(d):
    """
    Nonlinear function for spring force as defined in Equation (19).
    Gamma(d) = d + 0.75 if d <= -1
             = 0.25d   if -1 < d < 1
             = d - 0.75 if d >= 1
    """
    # Vectorized implementation for handling arrays
    if np.isscalar(d):
        if d <= -1:
            return d + 0.75
        elif d >= 1:
            return d - 0.75
        else:
            return 0.25 * d
    else:
        res = np.zeros_like(d)
        mask_neg = d <= -1
        mask_pos = d >= 1
        mask_mid = (~mask_neg) & (~mask_pos)
        
        res[mask_neg] = d[mask_neg] + 0.75
        res[mask_pos] = d[mask_pos] - 0.75
        res[mask_mid] = 0.25 * d[mask_mid]
        return res

# Generate input signal
T_end = 200  # End time (s)
dt_sim = 0.1  # Fixed time step for manual integration (s)

# Create a random piecewise constant signal
num_switches = 20
np.random.seed(42)  # Ensure reproducibility
switch_times = np.sort(np.random.uniform(0, T_end, num_switches))
switch_times = np.concatenate(([0], switch_times, [T_end]))
input_values = np.random.normal(0, 3, num_switches + 1)

def get_u(t):
    # Find the interval t falls into
    idx = np.searchsorted(switch_times, t, side='right') - 1
    idx = np.clip(idx, 0, len(input_values) - 1)
    return input_values[idx]

def dynamics(t, y):
    """
    State y = [x1, v1, x2, v2]
    """
    x1, v1, x2, v2 = y
    
    # Input force on mass 1
    u = get_u(t)
    
    # Displacements
    d1 = x1 # Spring 1 extension (Wall to M1)
    d2 = x2 - x1 # Spring 2 extension (M1 to M2)
    
    # Velocities difference
    dv1 = v1 # Relative velocity for damper 1
    dv2 = v2 - v1 # Relative velocity for damper 2
    
    # Spring Forces (Gamma non-linearity)
    F_s1 = k[0] * gamma(d1)
    F_s2 = k[1] * gamma(d2)
    
    # Damping Forces
    F_d1 = c[0] * dv1
    F_d2 = c[1] * dv2
    
    # Equations of motion
    # m1 * a1 = u - F_s1 - F_d1 + F_s2 + F_d2
    # m2 * a2 = - F_s2 - F_d2
    
    a1 = (u - F_s1 - F_d1 + F_s2 + F_d2) / m[0]
    a2 = (- F_s2 - F_d2) / m[1]
    
    return np.array([v1, a1, v2, a2])

def rk4_step(func, t, y, dt):
    """
    Performs one step of Runge-Kutta 4th Order.
    """
    k1 = func(t, y)
    k2 = func(t + 0.5 * dt, y + 0.5 * dt * k1)
    k3 = func(t + 0.5 * dt, y + 0.5 * dt * k2)
    k4 = func(t + dt, y + dt * k3)
    
    return y + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)

# Initial conditions [x1, v1, x2, v2]
y0 = np.array([0.0, 0.0, 0.0, 0.0])

print(f"Starting manual RK4 simulation with dt={dt_sim}s...")

# Simulation loop
t_vals = np.arange(0, T_end, dt_sim)
num_steps = len(t_vals)
y_vals = np.zeros((num_steps, 4))
y_vals[0] = y0

for i in range(1, num_steps):
    t_prev = t_vals[i-1]
    y_prev = y_vals[i-1]
    y_vals[i] = rk4_step(dynamics, t_prev, y_prev, dt_sim)

print("Simulation complete.")

# Plotting
plt.figure(figsize=(12, 8))

# Input
u_vals = [get_u(t) for t in t_vals]

plt.subplot(3, 1, 1)
plt.plot(t_vals, u_vals, label='Input Force u(t) (N)', color='red')
plt.ylabel('Force (N)')
plt.title('2-Spring Mass-Damper System Simulation (Manual RK4)')
plt.grid(True)
plt.legend()

# Position M1
plt.subplot(3, 1, 2)
plt.plot(t_vals, y_vals[:, 0], label='Position x1 (m)')
plt.ylabel('Position (m)')
plt.grid(True)
plt.legend()

# Position M2
plt.subplot(3, 1, 3)
plt.plot(t_vals, y_vals[:, 2], label='Position x2 (m)', color='green')
plt.ylabel('Position (m)')
plt.xlabel('Time (s)')
plt.grid(True)
plt.legend()

plt.tight_layout()
plt.savefig('simulation_2_springs_manual.png')
print("Plot saved to simulation_2_springs_manual.png")
