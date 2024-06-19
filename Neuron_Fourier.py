
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
    return np.fft.fft([z])

def Fourier_transform_derivative(f):
    f = np.fft.ifft(1j*np.fft.fft([f]))
    return f[0].real
    
# Function to calculate the output of the neuron
def neuron_output(inputs, weights, bias):
    # Calculate the weighted sum of inputs and their corresponding weights
    weighted_sum = np.dot(inputs, weights) + bias
    output = Fourier_transform(weighted_sum)
    return output

# Function to update the weights and bias using backpropagation
def update_weights(inputs, weights, bias, target_output, learning_rate):

    # Calculate the neuron's output
    output = neuron_output(inputs, weights, bias)[0]

    # Calculate the error (difference between the target output and the actual output)
    error = target_output - output

    # Calculate the derivative of the output with respect to the weighted sum (sigmoid derivative)
    d_output_d_weighted_sum = output * (1 - Fourier_transform_derivative(output))#(1 - output)

    # Calculate the gradient (chain rule)
    gradient = error * d_output_d_weighted_sum

    # Update each weight based on the error and gradient
    for i in range(len(weights)):
        # Calculate the partial derivative of the weighted sum with respect to the weight
        d_weighted_sum_d_weight = inputs[i]

        # Update the weight
        weights[i] += learning_rate * gradient * d_weighted_sum_d_weight

    # Update the bias (partial derivative of the weighted sum with respect to the bias is 1)
    bias += learning_rate * gradient

    return weights, bias

# Define the initial input values (example input features) and target output for training
training_inputs = [.2, .3, .4]

training_target_output = 2.5167

# Define new input values for testing (example new input features)
new_inputs = training_inputs#[2.0, -1.0, 0.5]
# Define the corresponding weights for each input (initially random or zero)
weights = [0.4, 0.2, -0.5]
# Define the bias term (initially random or zero)
bias = 0.2
# Define the learning rate (how quickly the model should learn)
learning_rate = 0.1

iterations = 150

for _ in range(iterations):
    weights, bias = update_weights(training_inputs, weights, bias, training_target_output, learning_rate)

# Calculate the output of the neuron with the new input data
new_output = neuron_output(new_inputs, weights, bias)[0]

# Print the final weights, bias, and output with the new input data
print("Iterations:", iterations)
print("Final weights:", weights)
print("Final bias:", bias)
print("Training target output:", training_target_output)
print("Output of the neuron: ", new_output)
