import numpy as np

def neuron(z):
    return np.fft.ifft(z).real
 
# Function to update the bias using backpropagation
def update_bias(inputs, bias, target_output):
    output = neuron(inputs + bias)     # Calculate the neuron's output
    for i in range(len(target_output)):
        bias[i] += (target_output[i] - output[i])  * output[i]  # Update the bias 
    return bias

# Define the initial input values (example input features) and target output for training
training_inputs = np.array([
[0.5, 0.7, 0.77], [0.8, 0.2, 0.432], [0.3, 0.9, 0.6754], [0.8, 0.1, 0.55543], [0.334, 0.6, 0.55543], [0.334, 0.1, 0.55543], [0.134, 0.1, 0.55543], 
[0.51, 0.7, 0.77],   [0.82, 0.2, 0.432],   [0.33, 0.9, 0.6754],    [0.84, 0.1, 0.55543],   [0.3345, 0.6, 0.55543],   [0.3346, 0.1, 0.55543],   [0.1347, 0.1, 0.55543],
[0.51, 0.71, 0.77], [0.82, 0.22, 0.432], [0.33, 0.93, 0.6754], [0.84, 0.14, 0.55543], [0.3345, 0.65, 0.55543], [0.3346, 0.13, 0.55543], [0.1347, 0.14, 0.55543],
[0.51, 0.72, 0.77], [0.82, 0.23, 0.432], [0.33, 0.94, 0.6754], [0.84, 0.15, 0.55543], [0.3345, 0.66, 0.55543], [0.3346, 0.14, 0.55543], [0.1347, 0.15, 0.5553]])

training_target = np.array([0.9, 0.7, 0.4, 0.56, 0.7, 0.234, 0.3, 0.91, 0.72, 0.43, 0.564, 0.75, 0.2346, 0.32, 0.91, 0.72, 0.4, 
0.564, 0.74, 0.2345, 0.36, 0.917, 0.728, 0.439, 0.5641, 0.7512, 0.234613, 0.322])

index = 10
if(index >= len(training_target)):
    index = len(training_target)-1
    
# Define new input values for testing (example new input features)
new_inputs = training_inputs[index]
#new_inputs = training_inputs

# Define the bias term (initially random or zero)
bias = np.zeros_like(training_inputs) 

# Define the number of iterations
iterations = 500

for _ in range(iterations):
    bias = update_bias(training_inputs, bias, training_target)

print("Iterations:", iterations)
print("Training target: ", training_target[index])

# Calculate the output of the neuron with the new input data
new_output = neuron(new_inputs + bias)
print("Neuron output: ", new_output[index][0])



