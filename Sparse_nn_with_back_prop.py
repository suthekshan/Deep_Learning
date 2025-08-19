import numpy as np

class SparseNN:
    def __init__(self, lr=0.01):
        self.lr = lr  # learning rate

        # Random initialization of weights
        self.w_in_h1 = np.random.randn(4, 3) * 0.1
        self.mask_in_h1 = np.array([
            [1, 0, 0],
            [1, 1, 0],
            [0, 1, 1],
            [0, 0, 1]
        ])
        self.w_in_h1 = self.w_in_h1 * self.mask_in_h1

        self.w_h1_h2 = np.random.randn(3, 4) * 0.1
        self.mask_h1_h2 = np.array([
            [1, 1, 0, 0],
            [0, 1, 1, 0],
            [0, 0, 1, 1]
        ])

        self.w_h2_out = np.random.randn(1, 3) * 0.1
        self.mask_h2_out = np.array([[1, 1, 1]])
        self.w_h2_out = self.w_h2_out * self.mask_h2_out

        # Bias terms
        self.b_h1 = np.zeros((4,))
        self.b_h2 = np.zeros((3,))
        self.b_out = np.zeros((1,))

    def relu(self, x):
        return np.maximum(0, x)

    def relu_deriv(self, x):
        return (x > 0).astype(float)

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def sigmoid_deriv(self, x):
        s = self.sigmoid(x)
        return s * (1 - s)

    def forward(self, x):
        x = np.atleast_2d(x)  # (batch, 3)

        # Input -> Hidden Layer I
        self.z1 = x @ (self.w_in_h1).T + self.b_h1
        self.h1 = self.relu(self.z1)

        # Hidden Layer I -> Hidden Layer II
        self.z2 = self.h1 @ (self.w_h1_h2 * self.mask_h1_h2).T + self.b_h2
        self.h2 = self.relu(self.z2)

        # Hidden Layer II -> Output
        self.z3 = self.h2 @ (self.w_h2_out).T + self.b_out
        self.out = self.sigmoid(self.z3)

        return self.out

    def backward(self, x, y):
        m = x.shape[0]
        x = np.atleast_2d(x)
        y = np.atleast_2d(y)

        # ---- Output layer ----
        dz3 = (self.out - y) * self.sigmoid_deriv(self.z3)   # (batch, 1)
        dw_h2_out = (dz3.T @ self.h2) / m
        db_out = dz3.mean(axis=0)

        # ---- Hidden Layer II ----
        dh2 = dz3 @ (self.w_h2_out)       # (batch, 3)
        dz2 = dh2 * self.relu_deriv(self.z2)
        dw_h1_h2 = (dz2.T @ self.h1) / m
        db_h2 = dz2.mean(axis=0)

        # ---- Hidden Layer I ----
        dh1 = dz2 @ (self.w_h1_h2 * self.mask_h1_h2)         # (batch, 4)
        dz1 = dh1 * self.relu_deriv(self.z1)
        dw_in_h1 = (dz1.T @ x) / m
        db_h1 = dz1.mean(axis=0)

        # ---- Update weights ----
        self.w_h2_out -= self.lr * dw_h2_out
        self.b_out -= self.lr * db_out

        self.w_h1_h2 -= self.lr * dw_h1_h2
        self.b_h2 -= self.lr * db_h2

        self.w_in_h1 -= self.lr * dw_in_h1
        self.b_h1 -= self.lr * db_h1

    def train(self, X, Y, epochs=1000):
        losses = []
        for epoch in range(epochs):
            out = self.forward(X)
            loss = np.mean((out - Y)**2)  # MSE
            losses.append(loss)
            self.backward(X, Y)

            if epoch % 100 == 0:
                print(f"Epoch {epoch}, Loss: {loss:.4f}")
        return losses


# ==== Generate dataset ====
np.random.seed(42)
n_samples = 200

X = np.random.randn(n_samples, 3)  # 3 input features
# Define target rule: y = 1 if x1 + 2*x2 - x3 > 0 else 0
Y = ((X[:, 0] + 2*X[:, 1] - X[:, 2]) > 0).astype(float).reshape(-1, 1)

# ==== Train model ====
model = SparseNN(lr=0.1)
losses = model.train(X, Y, epochs=1000)

print("\nFinal output (first 10 predictions):")
print(model.forward(X[:10]))
print("Targets (first 10):")
print(Y[:10])
