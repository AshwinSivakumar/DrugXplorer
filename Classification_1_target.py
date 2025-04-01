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
from sklearn.metrics import accuracy_score, classification_report
from sklearn.metrics import confusion_matrix
import seaborn as sns
import random
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader, random_split
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, matthews_corrcoef


# In[2]:


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

# Example usage
x = 'CC1C2C(CC3C2(CCC4C3CCC5C4(CCC(C5)OC6C(C(C(C(O6)CO)OC7C(C(C(C(O7)C)O)O)O)O)OC8C(C(C(C(O8)CO)O)O)O)C)C)OC1(CCC(C)COC9C(C(C(C(O9)CO)O)O)O)O'
new_bit_string = compare_fingerprints(x)
print(new_bit_string)


# In[18]:


df = pd.read_csv("CYP3A4_MORGAN.csv")
df


# In[19]:


# Remove rows where Y == 2
df = df[df['Y'] != 2].reset_index(drop=True)
df


# In[20]:


counts = df['Y'].value_counts()
print(counts)


# In[21]:


def set_seed(seed):
    # Set seed for PyTorch
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    
    # Set seed for NumPy
    np.random.seed(seed)
    
    # Set seed for Python's random module
    random.seed(seed)

    # Set CuDNN backend to deterministic for reproducibility on GPUs
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# Define your seed value
seed_value = 42
set_seed(seed_value)


# In[14]:


X = df.drop(columns=['Drug','Y']).values
y = df['Y'].values
X_tensor = torch.tensor(X, dtype=torch.float32)
y_tensor = torch.tensor(y, dtype=torch.long)  # Use long for classification labels

# Move tensors to the appropriate device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Split into train (75%) and test (25%) sets
X_train, X_test, y_train, y_test = train_test_split(X_tensor, y_tensor, test_size=0.25, random_state=42, stratify=y_tensor)

# Move to device
X_train, X_test = X_train.to(device), X_test.to(device)
y_train, y_test = y_train.to(device), y_test.to(device)

# Create DataLoaders
train_dataset = TensorDataset(X_train, y_train)
test_dataset = TensorDataset(X_test, y_test)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

# Define the classification model
class ClassificationModel(nn.Module):
    def __init__(self, input_dim=150, num_classes=3):
        super(ClassificationModel, self).__init__()
        self.fc1 = nn.Linear(input_dim, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, 256)
        self.fc4 = nn.Linear(256, 128)
        self.fc5 = nn.Linear(128, num_classes)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.relu(self.fc3(x))
        x = self.relu(self.fc4(x))
        x = self.fc5(x)  # No activation for logits
        return x

# Instantiate the model, loss function, and optimizer
model = ClassificationModel(input_dim=X_train.shape[1]).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training the model
num_epochs = 100

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    y_train_true, y_train_pred = [], []

    for batch_x, batch_y in train_loader:
        optimizer.zero_grad()
        outputs = model(batch_x)
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * batch_x.size(0)

        _, predictions = torch.max(outputs, 1)
        y_train_true.extend(batch_y.cpu().numpy())
        y_train_pred.extend(predictions.cpu().numpy())

    train_loss = running_loss / len(train_dataset)
    train_accuracy = accuracy_score(y_train_true, y_train_pred)

    # Evaluate on the test set after every epoch
    model.eval()
    test_loss = 0.0
    y_test_true, y_test_pred = [], []

    with torch.no_grad():
        for batch_x, batch_y in test_loader:
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            test_loss += loss.item() * batch_x.size(0)

            _, predictions = torch.max(outputs, 1)
            y_test_true.extend(batch_y.cpu().numpy())
            y_test_pred.extend(predictions.cpu().numpy())

    test_loss /= len(test_dataset)
    test_accuracy = accuracy_score(y_test_true, y_test_pred)

    print(f'Epoch [{epoch+1}/{num_epochs}], '
          f'Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.4f}, '
          f'Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}')

# Save the trained model
torch.save(model.state_dict(), 'CYP2D6_model_2.pth')
print('Model saved')

# Final evaluation on test set

# Final evaluation on test set
model.eval()
y_true, y_pred, y_probs = [], [], []

with torch.no_grad():
    for batch_x, batch_y in test_loader:
        outputs = model(batch_x)
        _, predictions = torch.max(outputs, 1)
        
        y_true.extend(batch_y.cpu().numpy())
        y_pred.extend(predictions.cpu().numpy())
        
        # Convert logits to probabilities using softmax
        probs = torch.softmax(outputs, dim=1)[:, 1]  # Probability of class 1
        y_probs.extend(probs.cpu().numpy())

# Compute metrics
accuracy = accuracy_score(y_true, y_pred)
precision = precision_score(y_true, y_pred, average='binary', zero_division=0)
recall = recall_score(y_true, y_pred, average='binary', zero_division=0)
f1 = f1_score(y_true, y_pred, average='binary')
mcc = matthews_corrcoef(y_true, y_pred)

# Compute AUC (only if both classes are present)
try:
    auc = roc_auc_score(y_true, y_probs)
except ValueError:
    auc = None

# Print results
print("\nFinal Test Performance:")
print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1-score: {f1:.4f}")
print(f"MCC: {mcc:.4f}")

if auc is not None:
    print(f"AUC: {auc:.4f}")
else:
    print("AUC computation failed (Ensure both classes have positive samples).")

print(classification_report(y_true, y_pred, target_names=['Inactive (0)', 'Active (1)']))


# In[19]:


import torch
import torch.nn as nn
from rdkit import Chem
from rdkit.Chem import AllChem,Draw
from rdkit import DataStructs
from rdkit.Chem import rdFingerprintGenerator
import numpy as np
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
class ClassificationModel(nn.Module):
    def __init__(self, input_dim=150, num_classes=3):
        super(ClassificationModel, self).__init__()
        self.fc1 = nn.Linear(input_dim, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, 256)
        self.fc4 = nn.Linear(256, 128)
        self.fc5 = nn.Linear(128, num_classes)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.relu(self.fc3(x))
        x = self.relu(self.fc4(x))
        x = self.fc5(x)  # No activation for logits
        return x
# Convert bit string to tensor
def bit_string_to_tensor(bit_string, n_bits=150):
    bit_list = [int(bit) for bit in bit_string]
    bit_tensor = torch.tensor(bit_list, dtype=torch.float32).unsqueeze(0)  # Add batch dimension
    return bit_tensor

# Define the function to predict using the trained model
def predict_y(model, smiles1, n_bits):
    # Assuming `compare_fingerprints` is a function that returns a bit string of length n_bits
    bit_string = compare_fingerprints(smiles1, nBits=n_bits)
    bit_tensor = bit_string_to_tensor(bit_string, n_bits)
    
    model.eval()
    with torch.no_grad():
        output = model(bit_tensor)
        predicted_class = torch.argmax(output, dim=1).item()  # Get the index of the highest score
    
   
    
    return predicted_class

# Example SMILES strings for testing
#P ='CC(C)/C=C/CCCCC(=O)NCC1=CC(=C(C=C1)O)OC'
P='CN1C=NC2=C1C(=O)N(C(=O)N2C)C'
#P='C1=CC(=CC=C1C(=O)CSC2=NC(=NC3=C2NC=N3)N)Br'
# Load the trained classification model
model = ClassificationModel(input_dim=150)  # Ensure input_dim matches your model's input size
model.load_state_dict(torch.load('CYP2D6_model.pth'))
model.eval()

# Make prediction
predicted_y = predict_y(model, P,150)
print(P)
print(f'Predicted Y for the test molecules: {predicted_y}')


# In[ ]:




