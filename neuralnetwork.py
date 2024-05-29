import numpy as np

# Define the sigmoid and its derivative functions
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)

# Define the NeuralNetwork class
class NeuralNetwork:
    def __init__(self, x, y):
        self.input = x
        self.weights1 = np.random.rand(self.input.shape[1], 4)
        self.weights2 = np.random.rand(4, 1)
        self.y = y
        self.output = np.zeros(self.y.shape)

    def feedforward(self):
        self.layer1 = sigmoid(np.dot(self.input, self.weights1))
        self.output = sigmoid(np.dot(self.layer1, self.weights2))

    def backprop(self):
        # Calculate the error
        error = self.y - self.output
        # Calculate the delta for weights2
        delta_weights2 = np.dot(self.layer1.T, error * sigmoid_derivative(self.output))
        # Calculate the delta for weights1
        delta_weights1 = np.dot(self.input.T, (np.dot(error * sigmoid_derivative(self.output), self.weights2.T) * sigmoid_derivative(self.layer1)))

        # Update the weights
        self.weights1 += delta_weights1
        self.weights2 += delta_weights2

    def train(self, iterations):
        for _ in range(iterations):
            self.feedforward()
            self.backprop()

# Define the input data (X) and output data (Y)
x = np.array([[0, 0, 1],
              [0, 1, 1],
              [1, 0, 1],
              [1, 1, 1]])

y = np.array([[0],
              [1],
              [1],
              [0]])

# Create the neural network
nn = NeuralNetwork(x, y)

# Train the neural network for 10,000 iterations
nn.train(50000)

# Print the final predicted output
print(nn.output)