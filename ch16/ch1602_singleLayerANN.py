import numpy as np
import matplotlib.pyplot as plt
import neurolab as nl
#load data
#input_data = np.loadtxt('./ch1602_neural_simple.txt')
input_data = np.array ([[2., 4., 0., 0.],
      [1.5, 3.9, 0., 0.],
      [2.2, 4.1, 0., 0.],
      [1.9, 4.7, 0., 0.],
      [5.4, 2.2, 0., 1.],
      [4.3, 7.1, 0., 1.],
      [5.8, 4.9, 0., 1.],
      [6.5, 3.2, 0., 1.],
      [3. , 2. , 1., 0.],
      [2.5, 0.5, 1., 0.],
      [3.5, 2.1, 1., 0.],
      [2.9, 0.3, 1., 0.],
      [6.5, 8.3, 1., 1.],
      [3.2, 6.2, 1., 1.],
      [4.9, 7.8, 1., 1.],
      [2.1, 4.8, 1., 1.]])
#print ('input_data: ', input_data)
# the first two column are features
# the last two columns are labels
data = input_data[:, 0:2]
#print ('data: ', data)
labels = input_data[:, 2:]
#print ('labels: ', labels)
# plot input data
plt.figure()
plt.scatter(data[:,0], data[:,1])
plt.xlabel('Dimension 1')
plt.ylabel('Dimension 2')
plt.title('Input data') 
# define min and max
dim1_min, dim1_max = data[:,0].min(), data[:,0].max()
dim2_min, dim2_max = data[:,1].min(), data[:,1].max()
#define number of neuron in output layer
nn_output_layer = labels.shape[1]
#define single-layer ANN
dim1 = [dim1_min, dim1_max]
dim2 = [dim2_min, dim2_max]
neural_net = nl.net.newp([dim1, dim2], nn_output_layer)
# print test result
print('\nTest Results:')
data_test = [[1.5, 3.2], [3.6, 1.7], [3.6, 5.7],[1.6, 3.9]]
for item in data_test:
    print(item, '-->', neural_net.sim([item])[0])
# Train the neural network with number of epochs
error = neural_net.train(data, labels, epochs = 200, show = 20, lr = 0.01)
# Plot data
plt.figure()
plt.plot(error)
plt.xlabel('Number of epochs')
plt.ylabel('Training error')
plt.title('Training error progress')
plt.grid()
plt.show()