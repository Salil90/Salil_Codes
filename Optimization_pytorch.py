import torch
import matplotlib.pyplot as plt

# Step 1: Set random seed for reproducibility
torch.manual_seed(42)

# Step 2: Generate synthetic data
X = torch.randn(100, 2)  # features: house size, number of bedrooms
true_weights = torch.tensor([[3.5], [2.0]])  # true weights for house size and bedrooms
true_bias = 1.0
y = torch.matmul(X, true_weights) + true_bias + torch.randn(100, 1) * 0.1  # true linear relation + noise

# Plotting y vs. the first feature (house size)
plt.scatter(X[:, 0], y)
plt.xlabel('House Size')
plt.ylabel('House Price')
plt.title('House Price vs. House Size')
plt.grid(True)
plt.show()

# Step 3: Define model architecture for polynomial regression
class PolynomialRegressionModel(torch.nn.Module):
    def __init__(self):
        super(PolynomialRegressionModel, self).__init__()
        self.linear = torch.nn.Linear(6, 1)  # input dimension: 6, output dimension: 1

    def forward(self, x):
        return self.linear(x)

# Step 4: Preprocess features for polynomial regression
X_poly = torch.cat((X, X**2, X**3), dim=1)  # add quadratic and cubic features

# Step 5: Instantiate the model
model = PolynomialRegressionModel()

# Step 6: Define loss function and optimizer
criterion = torch.nn.MSELoss()  # Mean Squared Error loss
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)  # Stochastic Gradient Descent

# Step 7: Training loop
num_epochs = 100
losses = []

for epoch in range(num_epochs):
    # Forward pass
    outputs = model(X_poly)
    loss = criterion(outputs, y)
    losses.append(loss.item())
    
    # Backward pass and optimization
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

# Step 8: Plot the loss curve
plt.plot(losses)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Loss Curve')
plt.grid(True)
plt.show()

# Step 9: Print the learned weights and bias
print('Learned weights:', model.linear.weight)
print('Learned bias:', model.linear.bias)
