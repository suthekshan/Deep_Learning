import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class SparseNN(nn.Module):
    def __init__(self):
        super(SparseNN, self).__init__()

        # Input (3) -> Hidden Layer I (4)
        self.w_in_h1 = nn.Parameter(torch.randn(4, 3) * 0.1)
        self.mask_in_h1 = torch.tensor([
            [1, 0, 0],  
            [1, 1, 0],  
            [0, 1, 1],  
            [0, 0, 1]   
        ], dtype=torch.float32)

        # Hidden Layer I (4) -> Hidden Layer II (3)
        self.w_h1_h2 = nn.Parameter(torch.randn(3, 4) * 0.1)
        self.mask_h1_h2 = torch.tensor([
            [1, 1, 0, 0],  
            [0, 1, 1, 0],  
            [0, 0, 1, 1]   
        ], dtype=torch.float32)

        # Hidden Layer II (3) -> Output (1)
        self.w_h2_out = nn.Parameter(torch.randn(1, 3) * 0.1)
        self.mask_h2_out = torch.tensor([[1, 1, 1]], dtype=torch.float32)

        # Bias terms
        self.b_h1 = nn.Parameter(torch.zeros(4))
        self.b_h2 = nn.Parameter(torch.zeros(3))
        self.b_out = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        # Input -> Hidden Layer I
        h1 = F.relu(F.linear(x, self.w_in_h1 * self.mask_in_h1, self.b_h1))

        # Hidden Layer I -> Hidden Layer II
        h2 = F.relu(F.linear(h1, self.w_h1_h2 * self.mask_h1_h2, self.b_h2))

        # Hidden Layer II -> Output
        out = torch.sigmoid(F.linear(h2, self.w_h2_out * self.mask_h2_out, self.b_out))

        return out


# ==== Example usage with training ====
torch.manual_seed(42)
model = SparseNN()

# Dummy dataset (10 samples, 3 features -> binary labels)
X = torch.randn(10, 3)
Y = (torch.rand(10, 1) > 0.5).float()  # random 0/1 labels

# Loss function (Binary Cross Entropy)
criterion = nn.BCELoss()
# Optimizer (Stochastic Gradient Descent)
optimizer = optim.SGD(model.parameters(), lr=0.1)
# Training loop
epochs = 500
for epoch in range(epochs):
    # Forward pass
    outputs = model(X)
    loss = criterion(outputs, Y)

    # Backward pass
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if epoch % 100 == 0:
        print(f"Epoch {epoch}, Loss: {loss.item():.4f}")
print("\nFinal predictions:")
print(model(X))
print("Targets:")
print(Y)
