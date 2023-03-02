import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

training_inputs = np.array([[0,0,1],
                            [0,1,1],
                            [1,0,1],
                            [1,1,0]])

training_outputs = np.array([[1,1,1,1]]).T

synaptic_weight = 2 * np.random.random((3,1)) - 1


for i in range(20000):
    input_layer = training_inputs
    outputs = sigmoid( np.dot(input_layer, synaptic_weight))
    err = training_outputs - outputs
    adjustments = np.dot( input_layer.T, err * (outputs * (1 - outputs)))
    synaptic_weight += adjustments
new = np.array([1, 0, 0])
outputs = sigmoid( np.dot(new, synaptic_weight))
print(outputs)