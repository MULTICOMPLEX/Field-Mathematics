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
    
def IFFT(z):
    return np.fft.ifft(z).real

def  FFT(f):
    f = np.fft.fft(f)
    return f.real 
    
# Function to calculate the output of the neuron
def neuron_output(inputs, bias):
    output = IFFT(inputs + bias)
    return output

# Function to update the weights and bias using backpropagation
def update_bias(inputs, bias, target_output, learning_rate):

    # Calculate the neuron's output
    output = neuron_output(inputs, bias)
    fft =  FFT(output)
    
    for i in range(len(target_output)):
    # Calculate the error (difference between the target output and the actual output)
        error = target_output[i] - output[i]
    # Calculate the gradient (chain rule)
        gradient = error * output[i] * (1 - fft[i])
    # Update the bias (partial derivative of the weighted sum with respect to the bias is 1)
        bias[i] += learning_rate * gradient

    return bias

# Define the initial input values (example input features) and target output for training
training_inputs = np.array([[0.5, 0.7, 0.77], [0.8, 0.2, 0.432], [0.3, 0.9, 0.6754], [0.8, 0.1, 0.55543], [0.334, 0.6, 0.55543], [0.334, 0.1, 0.55543]])
training_target = np.array([0.9, 0.7, 0.4, 0.56, 0.7, 0.234])

# Define new input values for testing (example new input features)
new_inputs = np.array([0.334, 0.1, 0.55543])
#new_inputs = training_inputs

# Define the bias term (initially random or zero)
bias = np.zeros_like(training_inputs)

# Define the number of iterations
iterations = 200

# Define the learning rate (how quickly the model should learn)
learning_rate = 4

for _ in range(iterations):
    bias = update_bias(training_inputs, bias, training_target, learning_rate)

print("Iterations:", iterations)
print("Learning rate:", learning_rate)
# Print the training_target
print("Training target: ", training_target)

# Calculate the output of the neuron with the new input data
new_output = neuron_output(new_inputs, bias)

first_column = []
for row in new_output:
    first_column.append(row[0])  # Extract the first element from each row
print("Neuron output: ", first_column) 
#print("Output of the neuron: ", new_output)
