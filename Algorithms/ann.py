import re
from joblib.numpy_pickle_utils import xrange
from numpy import *
import matplotlib.pyplot as plt


class NeuralNet(object):
 def __init__(self):
  # Generate random numbers
  random.seed(1)
  # Assigning random weights to a 3 x 1 matrix,
  self.synaptic_weights = 2 * random.random((3, 1)) - 1

 # The Sigmoid function
 def __sigmoid(self, x):
  return 1 / (1 + exp(-x))

 # The derivative of the Sigmoid function.
 def __sigmoid_derivative(self, x):
  return x * (1 - x)

 # Train the neural network and adjust the weights each time.
 def train(self, inputs, outputs, training_iterations):
  for iteration in xrange(training_iterations):
   # Pass the training set through the network.
   output = self.learn(inputs)

   # Calculate the error
   error = outputs - output

   # Adjust the weights by a factor
   factor = dot(inputs.T, error * self.__sigmoid_derivative(output))
   self.synaptic_weights += factor

  # The neural network thinks.

 def learn(self, inputs):
  return self.__sigmoid(dot(inputs, self.synaptic_weights))

def basicANN(inputval):
 # Initialize
 neural_network = NeuralNet()

 # The training set.
 inputs = array([[0, 1, 1], [1, 0, 0], [1, 0, 1]])
 outputs = array([[1, 0, 1]]).T

 itr = [10,100,1000,10000]    
 errorRate = []
 res =""
 # Train the neural network
 print(inputval)
 for i in itr:
    neural_network.train(inputs, outputs, i)
    errorRate.append(1-neural_network.learn(array(inputval)))
    res +="\nIteration : " +  str(i)
    res +="\nError Rate : " +str(errorRate[-1])

 # Test the neural network with a test example.
 plt.plot(itr, errorRate)
 plt.xlabel('Iterations ')
 plt.ylabel('Error Rate')
 plt.title('ANN')
 plt.show()
 print(neural_network.learn(array([1, 0, 1])))
 return res

