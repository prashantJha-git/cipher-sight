import numpy as np

# DEFINE NEURAL NETWORK
class NeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        self.weights_input_hidden = np.random.randn(
            self.input_size, self.hidden_size
        )
        self.weights_hidden_output = np.random.randn(
            self.hidden_size, self.output_size
        )
        self.bias_hidden = np.zeros((1, self.hidden_size))
        self.bias_output = np.zeros((1, self.output_size))

    def sigmoid(self, x):
        return 1/(1+np.exp(-x))

    def sigmoid_derivative(self, x):
        return x*(1-x)

    # DEFINE FEED FORWARD NETWORK
    def feedForward(self, x):
        self.hidden_activation = np.dot(
            x, self.weights_input_hidden
        )+self.bias_hidden
        self.hidden_output = self.sigmoid(self.hidden_activation)

        self.output_activation = np.dot(
            self.hidden_output, self.weights_hidden_output
        )+self.bias_output

        self.predicted_output = self.sigmoid(self.output_activation)
        return self.predicted_output

    # DEFINE BACKWARD NETWORK
    def backward(self, X, y, learning_rate):
        output_error = y-self.predicted_output
        output_delta = output_error*self.sigmoid_derivative(self.predicted_output)

        hidden_error = np.dot(output_delta, self.weights_hidden_output.T)
        hidden_delta = hidden_error*self.sigmoid_derivative(self.hidden_output)

        self.weights_hidden_output += np.dot(self.hidden_output.T, output_delta)*learning_rate
        self.bias_output += np.sum(output_delta, axis = 0, keepdims = True)*learning_rate
        self.weights_input_hidden += np.dot(X.T, hidden_delta)*learning_rate
        self.bias_hidden += np.sum(hidden_delta, axis = 0, keepdims = True)*learning_rate

    # TRAINING NEURAL NETWORK
    def train(self, X, y, epochs, learning_rate):
        for epoch in epochs:
            output = self.feedForward(X)
            self.backward(X, y, learning_rate)
            if epoch%5000==0:
                loss = np.mean(np.square(y-output))
                print(f"epoch:{epoch}, LOSS:{loss}")
