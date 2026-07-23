import numpy as np
import random

class Neuron():
    def __init__(self, nin):
        self.w = np.random.randn(nin) / np.sqrt(nin)
        self.b = random.uniform(-1, 1)
        self.grad_w = np.zeros(nin, dtype=float)
        self.grad_b = 0

    def _forward(self, X: np.ndarray):
        return self.w @ X + self.b

class Layer():
    def __init__(self, nin, nout):
        self.neurons = [Neuron(nin) for _ in range(nout)]
        self._actfunc = lambda x: x
        self.actfunc_grad = 1

        # updated in forward pass
        self.input_x = np.zeros(nin, dtype=float)
        self.unnormalized_outputs = np.zeros(nout, dtype=float)

    def compute_grad(self, out_grad):
        grad = self.actfunc_grad * out_grad  # (dOut / dJ) * (dJ / wx+b (unnormalized outputs))

        curr_grad = np.zeros_like(self.input_x, dtype=float)

        for i, neuron in enumerate(self.neurons):
            neuron_grad = grad[i]

            neuron.grad_w += self.input_x * neuron_grad

            neuron.grad_b += neuron_grad

            curr_grad += neuron.w * neuron_grad
        
        return curr_grad

    def _forward(self, input_x):
        self.input_x = input_x

        for i in range(len(self.neurons)):
            neuron = self.neurons[i]
            unnormalized_output = neuron._forward(self.input_x)
            self.unnormalized_outputs[i] = unnormalized_output

        output = self._actfunc(self.unnormalized_outputs)
        return output

class MLP():
    def __init__(self, layer_dims: list[int]):
        self.layers : list[Layer] = []

        for i in range(1, len(layer_dims)):
            curr_layer = Layer(layer_dims[i - 1], layer_dims[i])
            self.layers.append(curr_layer)

    def forward(self, input_data):
        last_data = input_data
        for layer in self.layers:
            output = layer._forward(last_data)

            last_data = output

        return last_data

    def backward(self, first_grad):
        prev_grad = first_grad
        for i in range(len(self.layers) - 1, -1, -1):
            layer = self.layers[i]
            new_grad = layer.compute_grad(prev_grad)
            prev_grad = new_grad

        return prev_grad

    def zero_grad(self):
        for layer in self.layers:
            for neuron in layer.neurons:
                neuron.grad_w = np.zeros(len(neuron.grad_w), dtype=float)
                neuron.grad_b = 0

    def grad_descent(self):
        for layer in self.layers:
            for neuron in layer.neurons:
                neuron.w -= 0.001 * neuron.grad_w
                neuron.b -= 0.001 * neuron.grad_b


input_data = np.array([5.0, 8.0], dtype=float)
mlp = MLP([2, 4, 4, 1])
output_data = np.array([5.0], dtype=float)

# loop
for _ in range(50):
    out = mlp.forward(input_data)

    loss = out - output_data
    print("Curr Error:", loss)

    prev_grad = mlp.backward(loss)
    mlp.grad_descent()

    # upd zero grad
    mlp.zero_grad()