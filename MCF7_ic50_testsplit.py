#!/usr/bin/env python
# coding: utf-8

# In[1]:


from rdkit import Chem
from rdkit.Chem import AllChem,Draw
from rdkit import DataStructs
from rdkit.Chem import rdFingerprintGenerator
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sea
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.ensemble import RandomForestRegressor
import math
import random
from sklearn.preprocessing import MinMaxScaler
from scipy.stats import median_abs_deviation


# In[179]:


df = pd.read_csv("ROCK1.csv")


# In[180]:


df
check  = df
check = check.apply(pd.to_numeric, errors='coerce')
df


# In[181]:


def compare_fingerprints(smiles1, radius=6, nBits=150):
    # Create a MorganGenerator with the specified radius and bit size
    morgan_gen = rdFingerprintGenerator.GetMorganGenerator(radius=radius, fpSize=nBits)
    
    # Convert SMILES strings to RDKit molecules
    mol1 = Chem.MolFromSmiles(smiles1)
    # Generate Morgan fingerprints using MorganGenerator
    fp1 = morgan_gen.GetFingerprint(mol1)
    
    # Convert fingerprints to bit strings
    bit_str1 = fp1.ToBitString()
    
    # Create a new bit string based on similarity
    new_bit_string = bit_str1 
    
    return new_bit_string


# In[182]:


df = df.dropna(subset=['y'])
df


# In[183]:


df['y'].plot(kind='box')


# In[184]:


def remove_outliers(df, column):

    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1

    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    filtered_df = df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]

    return filtered_df
df = remove_outliers(df, "y")


# In[185]:


df['y'].plot(kind='box')


# In[186]:


len(df)


# In[19]:


import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
import numpy as np
import math

# Assuming 'check' DataFrame is already defined with the required columns
X = df.drop(columns=['Drug','y']).values
y = df['y'].values

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Convert to PyTorch tensors
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.float32).view(-1, 1)

# Determine the device (GPU or CPU)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')

# Move tensors to the appropriate device
X_train_tensor, y_train_tensor = X_train_tensor.to(device), y_train_tensor.to(device)
X_test_tensor, y_test_tensor = X_test_tensor.to(device), y_test_tensor.to(device)

# Create DataLoaders
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
test_dataset = TensorDataset(X_test_tensor, y_test_tensor)

train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False)

# Define the CNN regression model
class CNNRegressionModel(nn.Module):
    def __init__(self, input_dim=150):
        super(CNNRegressionModel, self).__init__()
        # Assumes input is (batch_size, channels, input_dim), here channels=1
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(16, 32, kernel_size=3, padding=1)
        self.conv3 = nn.Conv1d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool1d(kernel_size=2)
        self.relu = nn.ReLU()

        # Fully connected layers
        self.fc1 = nn.Linear(64 * (input_dim // 8), 512)
        self.fc2 = nn.Linear(512, 128)
        self.fc3 = nn.Linear(128, 1)

    def forward(self, x):
        x = x.unsqueeze(1)  # Add channel dimension
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = self.pool(self.relu(self.conv3(x)))
        x = x.view(x.size(0), -1)  # Flatten for fully connected layers
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# Instantiate the model, loss function, and optimizer
model = CNNRegressionModel(input_dim=X_train_tensor.shape[1]).to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001)

# Training the model
# Training the model
num_epochs = 1000
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for batch_x, batch_y in train_loader:
        batch_x, batch_y = batch_x.to(device), batch_y.to(device)
        optimizer.zero_grad()
        outputs = model(batch_x)
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * batch_x.size(0)

    epoch_loss = running_loss / len(train_dataset)

    # Compute test loss after each epoch
    model.eval()
    test_loss = 0.0
    with torch.no_grad():
        for batch_x, batch_y in test_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            test_loss += loss.item() * batch_x.size(0)

    test_loss /= len(test_dataset)

    print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {epoch_loss:.4f}, Test Loss: {test_loss:.4f}')


# Save the trained model
torch.save(model.state_dict(), 'ESR1_MODEL.pth')
print('Model saved to cnn_regression_model.pth')

# Function to evaluate the model
def evaluate_model(loader, dataset_size):
    model.eval()
    predictions, actuals = [], []
    with torch.no_grad():
        for batch_x, batch_y in loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            outputs = model(batch_x)
            predictions.append(outputs.cpu().numpy())
            actuals.append(batch_y.cpu().numpy())

    predictions = np.vstack(predictions)
    actuals = np.vstack(actuals)

    # Calculate metrics
    mse = mean_squared_error(actuals, predictions)
    rmse = math.sqrt(mse)
    mae = mean_absolute_error(actuals, predictions)
    r2 = r2_score(actuals, predictions)

    return mse, rmse, mae, r2

# Evaluate on training and testing data
train_mse, train_rmse, train_mae, train_r2 = evaluate_model(train_loader, len(train_dataset))
test_mse, test_rmse, test_mae, test_r2 = evaluate_model(test_loader, len(test_dataset))

# Print metrics
print("\nTraining Metrics:")
print(f'Mean Squared Error (MSE): {train_mse:.4f}')
print(f'Root Mean Squared Error (RMSE): {train_rmse:.4f}')
print(f'Mean Absolute Error (MAE): {train_mae:.4f}')
print(f'R-squared (R²): {train_r2:.4f}')

print("\nTesting Metrics:")
print(f'Mean Squared Error (MSE): {test_mse:.4f}')
print(f'Root Mean Squared Error (RMSE): {test_rmse:.4f}')
print(f'Mean Absolute Error (MAE): {test_mae:.4f}')
print(f'R-squared (R²): {test_r2:.4f}')


# In[17]:


def bit_string_to_tensor(bit_string, n_bits=150):
    bit_list = [int(bit) for bit in bit_string]
    bit_tensor = torch.tensor(bit_list, dtype=torch.float32).unsqueeze(0)  # Add batch dimension
    return bit_tensor

# Define the function to predict using the trained model
def predict_y(model, smiles1, n_bits=150):
    bit_string = compare_fingerprints(smiles1, nBits=n_bits)
    bit_tensor = bit_string_to_tensor(bit_string, n_bits)
    
    model.eval()
    with torch.no_grad():
        prediction = model(bit_tensor)
    
    return prediction.item()

# Example SMILES strings for testing
P ='CC(C)/C=C/CCCCC(=O)NCC1=CC(=C(C=C1)O)OC' # Example SMILES string for testing
# Load the trained model
model = CNNRegressionModel(input_dim=150)  # Make sure input_dim matches your model's input size
model.load_state_dict(torch.load('PPARG_MODEL.pth'))
model.eval()

# Make prediction
predicted_y = predict_y(model, P)

print(f'Predicted Y for the test molecules: {predicted_y:.4f}')


# In[ ]:




