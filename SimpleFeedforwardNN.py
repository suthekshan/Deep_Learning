import numpy as np

class SimpleFeedforwardNN:
    def __init__(self, input_size, hidden_size, output_size):
        # Xavier/Glorot initialization for weights
        self.W1 = np.random.randn(hidden_size, input_size) * np.sqrt(2. / (input_size + hidden_size))
        self.b1 = np.zeros(hidden_size)
        self.W2 = np.random.randn(output_size, hidden_size) * np.sqrt(2. / (hidden_size + output_size))
        self.b2 = np.zeros(output_size)

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def relu(self, x):
        return np.maximum(0, x)

    def forward(self, x):
        # Input to hidden layer
        z1 = np.dot(self.W1, x) + self.b1
        a1 = self.relu(z1)
        # Hidden to output layer
        z2 = np.dot(self.W2, a1) + self.b2
        output = self.sigmoid(z2)
        return output

    def predict(self, x):
        output = self.forward(x)
        return 1 if output >= 0.5 else 0

# Example usage
input_size = 10      # Number of input features
hidden_size = 5      # Number of neurons in hidden layer
output_size = 1      # Single output for binary classification

model = SimpleFeedforwardNN(input_size, hidden_size, output_size)

sample_input = np.random.rand(input_size)    # Random input vector
output = model.forward(sample_input)

print(f"Input: {sample_input}")
print(f"Output (after sigmoid): {output}")
print(f"Predicted class: {model.predict(sample_input)}")
