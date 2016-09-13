import numpy as np


class BackPropagationNetwork(object):
    """
    A Back propagation network
    """
    # Class members
    layer_count = 0
    shape = None
    weights = []

    # Class Methods
    def __init__(self, layer_size):
        """
        Initialize the Network
        """
        # Layer info
        self.layer_count = len(layer_size) - 1
        self.layer_shape = layer_size

        # Input/Output data from run
        self._layer_input = []
        self._layer_output = []

        # Create the weight arrays
        for (l1, l2) in zip(layer_size[:-1], layer_size[1:]):
            self.weights.append(np.random.normal(scale=0.1, size=(l2, l1+1)))

    # Transfer function
    def sgm(self, x, derivative=False):
        out = 1 / (1 + np.exp(-x))
        if not derivative:
            return out
        return out * (1 - out)

    # Run Method
    def run(self, input):
        """
        Run the network based on the input data.
        """
        in_cases = input.shape[0]

        # Clear out the previous intermediate value lists
        self._layer_input = []
        self._layer_output = []

        print(input)
        # Run it!
        for index in range(self.layer_count):
            # Determine layer input
            if index == 0:
                print(input.T)
                print(np.ones([1, in_cases]))
                print(np.vstack([input.T, np.ones((1, in_cases))]))
                layer_input = self.weights[0].dot(np.vstack([input.T, np.ones([1, in_cases])]))
            else:
                layer_input = self.weights[index].dot(np.vstack([self._layer_output[-1], np.ones([1, in_cases])]))

            self._layer_input.append(layer_input)
            self._layer_output.append(self.sgm(layer_input))

        return self._layer_output[-1].T


# If run as a script, creates a test object
if __name__ == '__main__':
    bpn = BackPropagationNetwork((2, 2, 1))
    # print(bpn.weights)

    nn_input = np.array([[0, 0], [1, 1], [-1, 0.5]])
    nn_output = bpn.run(nn_input)
    # print("Input:\n{}\nOutput:\n{}".format(nn_input, nn_output))
