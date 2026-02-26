import numpy as np
import matplotlib.pyplot as plt
from simulate_2_springs_manual import y_vals, u_plot
# --- 1. Load Data (Simulating the Load Process) ---
# In a real workflow, you'd save Phase 1 data to .npz and load it here.
# For now, let's assume you just ran Phase 1 and have 'y_vals' and 'u_plot' available.
# If not, re-run the simulation block from Phase 1 first!

# y_vals shape is (N_steps, 4) -> [x1, v1, x2, v2]
# u_plot is list of length N_steps
# We need to transpose them to be (Features, Time)

X = y_vals.T  # Shape: (4, N)
U = np.array(u_plot).reshape(1, -1) # Shape: (1, N)

# --- 2. Create Training Matrices ---
# We want to predict x[k+1] using x[k] and u[k]
# X_next (Target): Columns 1 to End
X_next = X[:, 1:]

# X_curr (State): Columns 0 to End-1
X_curr = X[:, :-1]

# U_curr (Input): Columns 0 to End-1
U_curr = U[:, :-1]

# Stack State and Input for the "Predictor" matrix
# Z shape: (5, N-1) -> Top 4 rows are state, Bottom row is input
Z = np.vstack((X_curr, U_curr))

# --- 3. Solve Least Squares (The "Fit") ---
# We want to find Theta = [A, B] such that X_next approx Theta * Z
# Theta * Z = X_next  =>  Theta = X_next * pinv(Z)

print("Computing Least Squares fit...")
# standard least squares using numpy
# Theta_T, residuals, rank, s = np.linalg.lstsq(Z.T, X_next.T, rcond=None)
# Theta = Theta_T.T

# Or manually via Pseudo-Inverse (more explicit for control theory):
Z_pinv = np.linalg.pinv(Z)
Theta = X_next @ Z_pinv

# Extract A and B
A_fit = Theta[:, :4]  # First 4 columns
B_fit = Theta[:, 4:]  # Last column

print("\nIdentified A Matrix:")
print(np.round(A_fit, 3))
print("\nIdentified B Matrix:")
print(np.round(B_fit, 3))

# --- 4. Validation (Compare Linear Model vs Truth) ---
# Let's verify if this Linear Model is any good by simulating it
# starting from the same initial condition.

x_linear = np.zeros_like(X)
x_linear[:, 0] = X[:, 0]  # Same start

steps = X.shape[1]
for k in range(steps - 1):
    u_k = U[:, k]
    x_k = x_linear[:, k]
    # Apply the Linear Model: x_next = A*x + B*u
    x_linear[:, k+1] = A_fit @ x_k + (B_fit @ u_k)

# --- 5. Plot Comparison ---
plt.figure(figsize=(12, 6))

# Compare Position x1
plt.subplot(2, 1, 1)
plt.plot(X[0, :], 'k-', alpha=0.6, label='Ground Truth (Nonlinear)')
plt.plot(x_linear[0, :], 'r--', label='Linear Model (Prediction)')
plt.ylabel('Position x1 (m)')
plt.title('System Identification Results: Linear Fit vs Nonlinear Truth')
plt.legend()
plt.grid(True)

# Compare Position x2
plt.subplot(2, 1, 2)
plt.plot(X[2, :], 'k-', alpha=0.6, label='Ground Truth (Nonlinear)')
plt.plot(x_linear[2, :], 'g--', label='Linear Model (Prediction)')
plt.ylabel('Position x2 (m)')
plt.xlabel('Time Steps')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()

# Save matrices for later use in Phase 4
np.savez('linear_model.npz', A=A_fit, B=B_fit)
print("Linear model saved to linear_model.npz")