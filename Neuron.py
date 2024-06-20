import random
import numpy as np

def neuron1(inputs, weights, bias):
    """Calculates the output of a single neuron."""
    activation = bias
    for i in range(len(inputs)):
        activation += inputs[i] * weights[i]
    return 1 / (1 + np.e**-activation)

def neuron(inputs, weights, bias):
    """Calculates the output of a single neuron with ReLU activation."""
    activation = bias
    for i in range(len(inputs)):
        activation += inputs[i] * weights[i]
    return max(0, activation)  # ReLU activation

def train(inputs, target, weights, bias, learning_rate):
    """Trains the neuron using gradient descent with ReLU."""
    output = neuron(inputs, weights, bias)
    error = target - output

    # Backpropagation (weight update, adjusted for ReLU)
    relu_derivative = 1 if output > 0 else 0
    for i in range(len(weights)):
        weights[i] += learning_rate * error * relu_derivative * inputs[i]
    bias += learning_rate * error * relu_derivative
    return weights, bias, error

# Example training data
inputs = [[0.5, 0.7], [1.8, 0.2], [0.3, 0.9]]
targets = [0.9, 0.6, 0.4]

# Initialize weights and bias randomly
weights = [0.5 for _ in range(len(inputs[0]))]
bias = 0

learning_rate = 0.4
epochs = 100000
# Print frequency (adjust as needed)
print_every = 5000

for epoch in range(epochs):
    error_sum = 0
    for i in range(len(inputs)):
        weights, bias, error = train(inputs[i], targets[i], weights, bias, learning_rate)
        error_sum += error**2
        # Print training progress every 'print_every' epochs
    if (epoch + 1) % print_every == 0:
        print(f"Epoch {epoch + 1}: Error = {error_sum:.4f}")

# Test the trained neuron
test_input = [0.5, 0.7]
output = neuron(test_input, weights, bias)
print("Output for test input:", output)






