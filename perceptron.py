import numpy as np

class Perceptron():
    def __init__(self, input_size):
        self.weights = np.random.randn(input_size)
        self.bias = np.random.rand(1)

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def predict(self, inputs):
        #weighted sum 
        # dot product of input and weights 
        weighted_sum = np.dot(inputs, self.weights) + self.bias
        return self.sigmoid(weighted_sum)

# Example with input size 10
input_size = 10
model = Perceptron(input_size=input_size)

# Random input vector with 10 features
sample_input = np.random.rand(input_size)

#prediction
output = model.predict(sample_input)

print(f"Input: {sample_input}")
print(f"Output (after sigmoid): {output}")
print(f"Predicted class: {1 if output >= 0.5 else 0}")