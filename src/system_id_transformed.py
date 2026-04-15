import numpy as np
import matplotlib.pyplot as plt
import os
import sys

data_file = "transformed_simulation_data.npz"
if not os.path.exists(data_file):
    print(f"Error: {data_file} not found. Please generate it first.")
    sys.exit()

data = np.load(data_file)
# We now load z_vals instead of y_vals!
z_vals = data['z_vals'] 
u_plot = data['u_plot']

# convert to numpy arrays 
z_vals = np.array(z_vals)
u_plot = np.array(u_plot)

# define the bounds 
N = z_vals.shape[0]

# extract current states (z_k)
Z_curr = z_vals[:-1, :] 
# extract next states (z_{k+1})
Z_next = z_vals[1:, :] 
# extract inputs 
U_curr = u_plot[:-1] 
# extract next inputs 
U_next = u_plot[1:] 

# transpose all arrays 
Z_curr = Z_curr.T
Z_next = Z_next.T
U_curr = U_curr.reshape(1, -1)
U_next = U_next.reshape(1, -1)

# vertically stack the arrays to define the regression matrix Z_reg
# (Z_reg is used to avoid variable name conflict with z states)
Z_reg = np.vstack((Z_curr, U_next, U_curr))

# define: Z_next = theta * Z_reg
# calculate theta using pseudoinverse
theta = Z_next @ Z_reg.T @ np.linalg.pinv(Z_reg @ Z_reg.T)

# extract D, E_1, E_2 (Mapping z states, not x states!)
D = theta[:, :4]
E_1 = theta[:, 4:5]
E_2 = theta[:, 5:6]

z_pred = np.zeros_like(z_vals)
z_pred[0] = z_vals[0]

# the ARMA/Linear model to predict the next transformed state 
for k in range (N - 1):
    # current z state 
    z_k = z_pred[k].reshape(4,1)
    # current input 
    u_k = u_plot[k]
    u_k_next = u_plot[k+1]

    # calculate next z state 
    z_k_next = D @ z_k + E_1 * u_k_next + E_2 * u_k

    # store prediction 
    z_pred[k+1] = z_k_next.flatten()

# --- Plotting ---
plt.figure(figsize=(12, 8))

# Plot z1: Relative Position (x1 - x2)
plt.subplot(2, 1, 1)
plt.plot(z_vals[:, 0], 'k-', linewidth=2, label='Ground Truth (z1: x1 - x2)')
plt.plot(z_pred[:, 0], 'r--', linewidth=2, label='Linear Model Prediction')
plt.ylabel('Relative Position z1 (m)')
plt.title('Transformed Data Validation: Linear Model vs Ground Truth')
plt.grid(True)
plt.legend()

# Plot z3: Mass 2 Absolute Position (x2)
plt.subplot(2, 1, 2)
plt.plot(z_vals[:, 2], 'k-', linewidth=2, label='Ground Truth (z3: x2)')
plt.plot(z_pred[:, 2], 'g--', linewidth=2, label='Linear Model Prediction')
plt.ylabel('Mass 2 Position z3 (m)')
plt.xlabel('Time Steps')
plt.grid(True)
plt.legend()

plt.tight_layout()
plt.savefig("transformed_linear_model_comparison.png")
print("Saved transformed linear model comparison plot.")
plt.show()