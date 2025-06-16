import pandas as pd
import numpy as np
import tensorflow as tf
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


### Load and preprocess data ###

# Load the dataset
df = pd.read_csv('Salary.csv')
# print(df.head())

# Remove duplicates
df = df.drop_duplicates()

# Fill missing values (for numeric columns, use mean)
for col in df.select_dtypes(include=['float64', 'int64']).columns:
    df[col].fillna(df[col].mean(), inplace=True)

X = df[['YearsExperience']].values
y = df['Salary'].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale only the training data
scaler_X = StandardScaler()
scaler_y = StandardScaler()
X_train = scaler_X.fit_transform(X_train)
y_train = scaler_y.fit_transform(y_train.reshape(-1, 1))
X_test = scaler_X.transform(X_test)
y_test = scaler_y.transform(y_test.reshape(-1, 1))


### TensorFlow model ###

# Build a simple neural network
model_tf = tf.keras.Sequential([
    tf.keras.layers.Dense(10, activation='relu', input_shape=(1,)),
    tf.keras.layers.Dense(1)
])

model_tf.compile(optimizer='adam', loss='mse')
model_tf.fit(X_train, y_train, epochs=100, verbose=0)

# Evaluate
loss = model_tf.evaluate(X_test, y_test)
print(f"TensorFlow Test Loss: {loss}")


### PyTorch model ###

# Convert data to PyTorch tensors
X_train_torch = torch.FloatTensor(X_train)
y_train_torch = torch.FloatTensor(y_train).view(-1, 1)
X_test_torch = torch.FloatTensor(X_test)
y_test_torch = torch.FloatTensor(y_test).view(-1, 1)

# Defining a simple neural network
class SalaryModel(nn.Module):
    def __init__(self):
        super(SalaryModel, self).__init__()
        self.layer1 = nn.Linear(1, 10)
        self.layer2 = nn.Linear(10, 1)
    def forward(self, x):
        x = torch.relu(self.layer1(x))
        x = self.layer2(x)
        return x

model_pt = SalaryModel()
criterion = nn.MSELoss()
optimizer = optim.Adam(model_pt.parameters(), lr=0.01)

# Training loop
for epoch in range(100):
    optimizer.zero_grad()
    outputs = model_pt(X_train_torch)
    loss = criterion(outputs, y_train_torch)
    loss.backward()
    optimizer.step()

# Evaluate
model_pt.eval()
with torch.no_grad():
    y_pred_pt = model_pt(X_test_torch)
    loss_pt = criterion(y_pred_pt, y_test_torch)
    print(f"PyTorch Test Loss: {loss_pt.item()}")


### Visualize results ###

# Predict with TensorFlow
y_pred_tf = scaler_y.inverse_transform(model_tf.predict(X_test))


# Predict with PyTorch
with torch.no_grad():
    y_pred_pt = model_pt(X_test_torch)
y_pred_pt = scaler_y.inverse_transform(y_pred_pt.numpy())

plt.scatter(X_test, y_test, color='red', label='Actual')
plt.scatter(X_test, y_pred_tf, color='blue', label='TensorFlow')
plt.scatter(X_test, y_pred_pt, color='green', label='PyTorch')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.legend()
plt.show()
