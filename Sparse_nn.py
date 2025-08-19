import numpy as np

class SparseNN:
    def __init__(self):
        # Random initialization of weights
        self.w_in_h1 = np.random.randn(4, 3)
        self.mask_in_h1 = np.array([
            [1, 0, 0],  # Input1 -> Hidden1
            [1, 1, 0],  # Input1,2 -> Hidden2
            [0, 1, 1],  # Input2,3 -> Hidden3
            [0, 0, 1]   # Input3 -> Hidden4
        ])
        self.w_h1=self.w_in_h1 * self.mask_in_h1

        self.w_h1_h2 = np.random.randn(3, 4)
        self.mask_h1_h2 = np.array([
            [1, 1, 0, 0],  # H1->H2_1, H2->H2_1
            [0, 1, 1, 0],  # H2->H2_2, H3->H2_2
            [0, 0, 1, 1]   # H3->H2_3, H4->H2_3
        ])
        self.w_h2=self.w_h1_h2 * self.mask_h1_h2

        self.w_h2_out = np.random.randn(1, 3)
        self.mask_h2_out = np.array([[1, 1, 1]])
        self.w_h2_out=self.w_h2_out * self.mask_h2_out


        # Bias terms
        self.b_h1 = np.zeros((4,))
        self.b_h2 = np.zeros((3,))
        self.b_out = np.zeros((1,))

    def relu(self, x):
        return np.maximum(0, x)

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def forward(self, x):
        """
        x: shape (batch, 3) or (3,)
        """

        # Input -> Hidden Layer I
        h1 = self.relu(x @ (self.w_h1).T + self.b_h1)

        # Hidden Layer I -> Hidden Layer II
        h2 = self.relu(h1 @ (self.w_h2).T + self.b_h2)

        # Hidden Layer II -> Output
        out = self.sigmoid(h2 @ (self.w_h2_out).T + self.b_out)

        return out

# Example run with random datapoints
np.random.seed(42)
model = SparseNN()
x = np.random.randn(5, 3)  # batch of 5 datapoints, each with 3 features
y = model.forward(x)

print("Input:\n", x)
print("Output:\n", y)
