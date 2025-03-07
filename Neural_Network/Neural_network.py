import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
from scipy.stats import pearsonr
import os
import sys


# Load data
df_rna = pd.read_hdf('train_cite_inputs.h5')
df_y = pd.read_hdf('train_cite_targets.h5')

# Calculate variances and select the top 4096 features
variances = df_rna.var().sort_values(ascending=False)
top_features = variances.head(4096).index
X_filtered = df_rna[top_features].values
y_filtered = df_y.values

# Split data
X_train, X_test, y_train, y_test = train_test_split(X_filtered, y_filtered, test_size=0.2, random_state=42)

X_train, X_test, y_train, y_test = map(torch.tensor, (X_train, X_test, y_train, y_test))

# Create datasets and DataLoaders
train_dataset = TensorDataset(torch.FloatTensor(X_train), torch.FloatTensor(y_train))
train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)

# Define the Model
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fcs = nn.Sequential(
            nn.Linear(4096, 2048),
            nn.BatchNorm1d(2048),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(2048, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 140),
            nn.BatchNorm1d(140),
            nn.ReLU()
        )

    def forward(self, x):
        return self.fcs(x)

# Initialize Model, Loss Function, and Optimizer
model = Net()
criterion = nn.L1Loss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
criterion.to(device)

def load_checkpoint(model_dir='model_checkpoints'):
    checkpoints = [os.path.join(model_dir, f) for f in os.listdir(model_dir) if f.endswith('.pt')]
    if checkpoints:
        latest_checkpoint = max(checkpoints, key=os.path.getctime)
        print(f"Resuming from checkpoint: {latest_checkpoint}")
        data = torch.load(latest_checkpoint)
        model.load_state_dict(data['model_state_dict'])
        optimizer.load_state_dict(data['optimizer_state_dict'])
        return data['epoch'] + 1
    return 1

start_epoch = load_checkpoint()

def train(model, train_loader, criterion, optimizer, start_epoch, total_epochs=50, checkpoint_interval=5, model_dir='model_checkpoints'):
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    model.train()
    for epoch in range(start_epoch, total_epochs + 1):
        epoch_loss = 0
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        print(f"Epoch {epoch}/{total_epochs}, Loss: {epoch_loss / len(train_loader):.4f}")

        # Save checkpoint every 5 epochs
        if epoch % checkpoint_interval == 0:
            checkpoint_path = os.path.join(model_dir, f'model_epoch_{epoch}.pt')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': epoch_loss / len(train_loader)
            }, checkpoint_path)
            print(f"Model saved to {checkpoint_path}")

# Train the model
train(model, train_loader, criterion, optimizer, start_epoch)

# Evaluate model
model.eval()
X_test_tensor = torch.FloatTensor(X_test).to(device)
Y_test = y_test.numpy()

with torch.no_grad():
    Y_pred_tensor = model(X_test_tensor)
    Y_pred = Y_pred_tensor.cpu().numpy()

print(Y_test.shape,Y_pred.shape)
correlations = []
for i in range(Y_test.shape[1]):
    if np.std(Y_test[:, i]) == 0 or np.std(Y_pred[:, i]) == 0:
        correlations.append(-1.0)
    else:
        correlation, _ = pearsonr(Y_test[:, i], Y_pred[:, i])
        correlations.append(correlation)

avg_correlation = np.mean(correlations)
print(f"Final Test Set Pearson Correlation Score: {avg_correlation}")
