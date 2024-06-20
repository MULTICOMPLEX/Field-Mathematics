import numpy as np


# Import the exp function from the math module
import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def tanh(x):
    return np.tanh(x)

def relu(x):
    return np.maximum(0, x)

def leaky_relu(x, alpha=0.01):
    return np.where(x > 0, x, alpha * x)

def prelu(x, alpha=0.1):
    return np.where(x > 0, x, alpha * x)

def elu(x, alpha=1.0):
    return np.where(x > 0, x, alpha * (np.exp(x) - 1))

def swish(x):
    return x * sigmoid(x)
    
def Fourier_transform(z):
    return np.fft.fft(z).real

def Fourier_transform_derivative(f):
    f = np.fft.ifft(1j*np.fft.fft(f))
    return f.real 
    
# Function to calculate the output of the neuron
def neuron_output(inputs, bias):
    a = inputs + bias
    output = Fourier_transform(a)
    return output

# Function to update the weights and bias using backpropagation
def update_bias(inputs, bias, target_output, learning_rate):

    # Calculate the neuron's output
    output = neuron_output(inputs, bias)

    for i in range(len(target_output)):
    # Calculate the error (difference between the target output and the actual output)
        error = target_output[i] - output[i]
    # Calculate the gradient (chain rule)
        gradient = error * output[i] * (1 - Fourier_transform_derivative(output[i]))
    # Update the bias (partial derivative of the weighted sum with respect to the bias is 1)
        bias[i] += learning_rate * gradient

    return bias

# Define the initial input values (example input features) and target output for training
training_inputs = np.array([[0.5, 0.7], [0.8, 0.2], [0.3, 0.9]])
training_target_output = np.array([0.9, 0.7, 0.4])

# Define new input values for testing (example new input features)
new_inputs = np.array([0.8, 0.2])
new_inputs = training_inputs


# Define the bias term (initially random or zero)
bias = np.array([[0.0, 0.0],[0.0, 0.0],[0.0, 0.0]])
# Define the learning rate (how quickly the model should learn)
learning_rate = 0.3

iterations = 150

for _ in range(iterations):
    bias = update_bias(training_inputs, bias, training_target_output, learning_rate)

# Calculate the output of the neuron with the new input data
new_output = neuron_output(new_inputs, bias)

# Print the final weights, bias, and output with the new input data
print("Iterations:", iterations)
#print("Final bias:", bias)
print("Training target output:", training_target_output)

first_column = []
for row in new_output:
    first_column.append(row[0])  # Extract the first element from each row
print("Output of the neuron: ", first_column) 
#print("Output of the neuron: ", new_output)
