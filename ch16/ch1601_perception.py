# import ANN packages
import matplotlib.pyplot as plt
import neurolab as nl

# SUpervised Learning with input and target
input = [[0, 0], [0, 1], [1, 0], [1, 1]]
target = [[0], [0], [0], [1]]
# Create ANN with two input and one target
net = nl.net.newp([[0, 1],[0, 1]], 1)
# Train the ANN with delta rule
error_progress = net.train(input, target, epochs=100, show=10, lr=0.1)
# plot
plt.figure()
plt.plot(error_progress)
plt.xlabel('Number of epochs')
plt.ylabel('Training error')
plt.grid()
plt.show()