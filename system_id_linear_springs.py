import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import sys

data_file = "linear_simulation_data.npz"
if not os.path.exists(data_file):
    print(f"Error: {data_file} not found. Please run 'simulate_2_springs_simple.py' once first to generate the data.")
    sys.exit()

data = np.load(data_file)
y_vals = data['y_vals']
u_plot = data['u_plot']

# #convert to pandas dataframes 
# y_vals = pd.DataFrame(y_vals)
# u_plot = pd.DataFrame(u_plot)

# print(y_vals)
# print(u_plot)

# convert to numpy arrays 
y_vals = np.array(y_vals)
u_plot = np.array(u_plot)

# define the bounds 
N = y_vals.shape[0]

# extract current states 
X_curr = y_vals[:-1, :] 
# extract next states 
X_next = y_vals[1:, :] 
# extract inputs 
U_curr = u_plot[:-1] 
# extract next inputs 
U_next = u_plot[1:] 

# transpose all arrays 
X_curr = X_curr.T
X_next = X_next.T
U_curr = U_curr.reshape(1, -1)
U_next = U_next.reshape(1, -1)

# vertically stack the arrays to define z
Z = np.vstack((X_curr, U_next, U_curr))

# define: X_next = theta * Z
# theta = X_next * Z.T * (Z * Z.T)^-1
# calculate theta 
theta = X_next @ Z.T @ np.linalg.pinv(Z @ Z.T)

# extract D, E_1, E_2
D = theta[:, :4]
E_1 = theta[:, 4:5]
E_2 = theta[:, 5:6]

y_pred = np.zeros_like(y_vals)

y_pred[0] = y_vals[0]

# the ARMA model to predict the next state 
for k in range (N - 1):
    # current state 
    x_k = y_pred[k].reshape(4,1)
    # current input 
    u_k = u_plot[k]
    u_k_next = u_plot[k+1]

    #calculate next state 
    x_k_next = D @ x_k + E_1 * u_k_next + E_2 * u_k

    # store prediction 
    y_pred[k+1] = x_k_next.flatten()

plt.figure(figsize=(12, 8))

# Plot Cart 1 (Position x1)
plt.subplot(2, 1, 1)
plt.plot(y_vals[:, 0], 'k-', linewidth=2, label='Ground Truth (Nonlinear)')
plt.plot(y_pred[:, 0], 'r--', linewidth=2, label='Linear Model (D, E1, E2)')
plt.ylabel('Cart 1 Position (m)')
plt.title('System ID Validation: Linear Model vs Ground Truth')
plt.grid(True)
plt.legend()

# Plot Cart 2 (Position x2)
plt.subplot(2, 1, 2)
plt.plot(y_vals[:, 2], 'k-', linewidth=2, label='Ground Truth (Nonlinear)')
plt.plot(y_pred[:, 2], 'g--', linewidth=2, label='Linear Model (D, E1, E2)')
plt.ylabel('Cart 2 Position (m)')
plt.xlabel('Time Steps')
plt.grid(True)
plt.legend()

plt.tight_layout()
plt.savefig("linear_springs_model_comparison.png")
plt.show()