
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
    return np.fft.fft(z)

def Fourier_transform_derivative(f):
    f = np.fft.ifft(1j*p*np.fft.fft([f]))
    return f[0].real 
    
# Function to calculate the output of the neuron
def neuron_output(inputs, bias):
    a = inputs + bias
    output = Fourier_transform(a)
    return output[0].real 

# Function to update the weights and bias using backpropagation
def update_bias(inputs, bias, target_output, learning_rate):

    # Calculate the neuron's output
    output = neuron_output(inputs, bias)

    # Calculate the error (difference between the target output and the actual output)
    error = target_output - output

    # Calculate the gradient (chain rule)
    gradient = error * output * (1 - Fourier_transform_derivative(output))

    # Update the bias (partial derivative of the weighted sum with respect to the bias is 1)
    bias += learning_rate * gradient

    return bias

# Define the initial input values (example input features) and target output for training
training_inputs = np.array([.2, .3, .4, .7, .3, .3, -.36])

N = training_inputs.size
x = 2*np.pi*np.arange(0,N,1)/N #-open-periodic domain    
dx = x[1]-x[0]
p = np.fft.fftfreq(N, d = dx) * 2*np.pi 


training_target_output = 1.4123762

# Define new input values for testing (example new input features)
new_inputs = training_inputs
new_inputs[5] = 0.1
new_inputs[4] = 0.1
new_inputs[3] = 0
new_inputs[2] = 0
new_inputs[1] = 0


# Define the bias term (initially random or zero)
bias = 0.
# Define the learning rate (how quickly the model should learn)
learning_rate = 0.07

iterations = 100

for _ in range(iterations):
    bias = update_bias(training_inputs, bias, training_target_output, learning_rate)

# Calculate the output of the neuron with the new input data
new_output = neuron_output(new_inputs, bias)

# Print the final weights, bias, and output with the new input data
print("Iterations:", iterations)
print("Final bias:", bias)
print("Training target output:", training_target_output)
print("Output of the neuron: ", new_output)