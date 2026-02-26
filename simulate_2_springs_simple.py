import numpy as np
import matplotlib.pyplot as plt

# --- 1. System Parameters (Same as Paper) ---
m = np.array([1/4, 1/3])  # Masses (kg)
c = np.array([1/4, 1/3])  # Damping (Ns/m)
k = np.array([1.0, 5/6])  # Stiffness (N/m)

# --- 2. The Nonlinear Spring Function ---
def gamma(d):
    """ Nonlinear spring profile (Hardening Spring) """
    if np.isscalar(d):
        if d <= -1: return d + 0.75
        elif d >= 1: return d - 0.75
        else: return 0.25 * d
    else:
        # Vectorized version for arrays
        res = np.zeros_like(d)
        neg, pos = (d <= -1), (d >= 1)
        mid = (~neg) & (~pos)
        res[neg] = d[neg] + 0.75
        res[pos] = d[pos] - 0.75
        res[mid] = 0.25 * d[mid]
        return res

# --- 3. Input Signal Generation (FIXED) ---
T_end = 200.0        # Total time
dt_sim = 0.01        # Time step
avg_switch_time = 5.0 # Force a switch every 5 seconds

# Create Fixed Intervals (0, 5, 10, 15...)
switch_times = np.arange(0, T_end + avg_switch_time, avg_switch_time)
# Random amplitudes for each interval
np.random.seed(42)
input_values = np.random.normal(0, 2.0, len(switch_times)) 

def get_u(t):
    """ Returns the step input at time t """
    # Find which 5-second window we are in
    idx = int(t // avg_switch_time)
    idx = np.clip(idx, 0, len(input_values) - 1)
    return input_values[idx]

# --- 4. Physics Engine (Dynamics) ---
def dynamics(t, y):
    x1, v1, x2, v2 = y
    
    u = get_u(t)
    
    # Stretches
    d1 = x1
    d2 = x2 - x1
    
    # Velocities (for damping)
    dv1 = v1
    dv2 = v2 - v1
    
    # Forces
    F_s1 = k[0] * gamma(d1)
    F_s2 = k[1] * gamma(d2)
    F_d1 = c[0] * dv1
    F_d2 = c[1] * dv2
    
    # Accelerations (F = ma)
    # M1 feels U (push), F_s1/d1 (pull back), F_s2/d2 (pull forward from M2)
    a1 = (u - F_s1 - F_d1 + F_s2 + F_d2) / m[0]
    # M2 feels -F_s2/d2 (pull back from M1)
    a2 = (- F_s2 - F_d2) / m[1]
    
    return np.array([v1, a1, v2, a2])

# --- 5. Manual RK4 Solver ---
def rk4_step(func, t, y, dt):
    k1 = func(t, y)
    k2 = func(t + 0.5*dt, y + 0.5*dt*k1)
    k3 = func(t + 0.5*dt, y + 0.5*dt*k2)
    k4 = func(t + dt, y + dt*k3)
    return y + (dt/6.0)*(k1 + 2*k2 + 2*k3 + k4)

# --- 6. Run Simulation ---
t_vals = np.arange(0, T_end, dt_sim)
num_steps = len(t_vals)
y_vals = np.zeros((num_steps, 4)) # [x1, v1, x2, v2]

print(f"Simulating {T_end}s with 5s intervals...")
for i in range(1, num_steps):
    y_vals[i] = rk4_step(dynamics, t_vals[i-1], y_vals[i-1], dt_sim)

# --- 7. Plotting ---
plt.figure(figsize=(10, 8))

# Input Force (Verify it is steps, not ramps)
u_plot = [get_u(t) for t in t_vals]
plt.subplot(3, 1, 1)
plt.plot(t_vals, u_plot, color='red', label='Input u(t)')
plt.title("Input Force (Step Function)")
plt.ylabel("Force (N)")
plt.grid(True)
plt.legend()

# Position M1
plt.subplot(3, 1, 2)
plt.plot(t_vals, y_vals[:, 0], label='Cart 1 Position')
plt.ylabel("Position (m)")
plt.grid(True)
plt.legend()

# Position M2
plt.subplot(3, 1, 3)
plt.plot(t_vals, y_vals[:, 2], color='green', label='Cart 2 Position')
plt.xlabel("Time (s)")
plt.ylabel("Position (m)")
plt.grid(True)
plt.legend()

plt.tight_layout()
plt.show()