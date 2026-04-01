import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import os
import sys

# load the data 
data_file = "simulation_data.npz"
if not os.path.exists(data_file):
    print(f"Error: {data_file} not found. Please run 'simulate_2_springs_simple.py' once first to generate the data.")
    sys.exit()

data = np.load(data_file)
y_vals = data['y_vals']
u_plot = data['u_plot']

# normalize the data 
y_mean, y_std = y_vals.mean(axis=0), y_vals.std(axis=0)
u_mean, u_std = u_plot.mean(), u_plot.std()

y_norm = (y_vals - y_mean) / (y_std + 1e-8)
u_norm = (u_plot - u_mean) / (u_std + 1e-8)

# convert to torch tensors
Y = torch.tensor(y_norm, dtype=torch.float32)
U = torch.tensor(u_norm, dtype=torch.float32).unsqueeze(1)

# define the model
class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(RNN, self).__init__()
        self.hidden_size = hidden_size
        
        # Memory Update Weights
        self.W_hh = nn.Parameter(torch.randn(hidden_size, hidden_size) * 0.1)
        self.W_uh = nn.Parameter(torch.randn(input_size, hidden_size) * 0.1)
        self.b_h = nn.Parameter(torch.zeros(hidden_size))
        
        # State Prediction Weights
        self.W_hx = nn.Parameter(torch.randn(hidden_size, output_size) * 0.1)
        self.b_x = nn.Parameter(torch.zeros(output_size))
    
    def forward(self, u_seq):
        # get sequence length
        seq_len = u_seq.shape[0]

        # initialize hidden state
        h_k = torch.zeros(self.hidden_size)

        # initialize list to store predictions
        y_pred = []

        # loop through the sequence
        for k in range(seq_len):
            # get current input
            u_k = u_seq[k]

            # update hidden state
            h_k = torch.relu(h_k @ self.W_hh + u_k @ self.W_uh + self.b_h)

            # predict next state
            y_k = h_k @ self.W_hx + self.b_x

            # store prediction
            y_pred.append(y_k)

        # convert list to tensor
        y_pred = torch.stack(y_pred)

        return y_pred

# train the model
model = RNN(input_size=1, hidden_size=4, output_size=4)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# training loop
n_epochs = 3000
seq_len = 1000

for epoch in range(n_epochs):
    # zero the gradients
    optimizer.zero_grad()
    
    # Pick a random starting point in the sequence
    start_idx = np.random.randint(0, len(U) - seq_len)
    end_idx = start_idx + seq_len
    
    # Get the chunks of data using slicing [start_idx:end_idx]
    U_chunk = U[start_idx:end_idx]
    Y_chunk = Y[start_idx:end_idx]
    
    # Forward pass
    Y_pred_chunk = model(U_chunk)
    
    # Calculate Loss
    loss = criterion(Y_pred_chunk, Y_chunk)
    
    # Backpropagate and Update
    loss.backward()
    optimizer.step()
    
    # Print the loss every 10 epochs
    if (epoch + 1) % 10 == 0:
        print(f"Epoch {epoch+1}/{n_epochs}, Loss: {loss.item():.4f}")

# Full System Rollout
print("Evaluating trained RNN...")
with torch.no_grad():
    # Pass the ENTIRE sequence U through the model
    Y_pred_norm = model(U)
    # Convert to numpy and un-normalize
    Y_pred = Y_pred_norm.detach().numpy() * y_std + y_mean

# Plotting Results
plt.figure(figsize=(12, 8))

# Plot Cart 1 Position (x1)
plt.subplot(2, 1, 1)
plt.plot(y_vals[:, 0], 'k-', label='Ground Truth')
plt.plot(Y_pred[:, 0], 'r--', label='RNN Prediction')
plt.ylabel('Cart 1 Position (m)')
plt.legend()
plt.grid(True)

# Plot Cart 2 Position (x2)
plt.subplot(2, 1, 2)
plt.plot(y_vals[:, 2], 'k-', label='Ground Truth')
plt.plot(Y_pred[:, 2], 'g--', label='RNN Prediction')
plt.ylabel('Cart 2 Position (m)')
plt.xlabel('Time Steps')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.savefig("Improved RNN Comparison.png")
plt.show()

    