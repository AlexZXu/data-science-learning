import random
import numpy as np
import idx2numpy
import numpy as np
import matplotlib.pyplot as plt
import math

y_true : np.ndarray

"""
NN CLASSES AND BACKPROP GRADIENTS
"""

class Neuron():
    def __init__(self, nin):
        self.w = np.random.randn(nin) / np.sqrt(nin)
        self.b = random.uniform(-1, 1)
        self.grad_w = np.zeros(nin, dtype=float)
        self.grad_b = 0

    def _forward(self, X: np.ndarray):
        return self.w @ X + self.b

class Layer():
    def __init__(self, nin, nout, actfunc="linear"):
        self.neurons = [Neuron(nin) for _ in range(nout)]
        self.obtain_actfunc(actfunc)

        # updated in forward pass
        self.input_x = np.zeros(nin, dtype=float)
        self.unnormalized_outputs = np.zeros(nout, dtype=float)

    def obtain_actfunc(self, actfunc):
        if (actfunc == "linear"):
            self._actfunc = lambda x: x
            self.actfunc_grad = lambda x: np.ones_like(x)
        if (actfunc == "relu"):
            self._actfunc = lambda x: np.maximum(0, x)
            self.actfunc_grad = lambda x: (x > 0).astype(float)
        if (actfunc == "sigmoid"):
            def sigmoid(x):
                return 1.0 / (1.0 + np.exp(-x))
            self._actfunc = lambda x: sigmoid(x)
            self.actfunc_grad = lambda x: sigmoid(x) * (1.0 - sigmoid(x))
        if (actfunc == "tanh"):
            self._actfunc = lambda x: np.tanh(x)
            self.actfunc_grad = lambda x: 1.0 - np.tanh(x)**2
        if (actfunc == "softmax"):
            def softmax(x):
                shifted_x = x - np.max(x)
                exp_x = np.exp(shifted_x)
                return exp_x / np.sum(exp_x)
            self._actfunc = lambda x: softmax(x)
            self.actfunc_grad = lambda x: softmax(x) - y_true

    def compute_grad(self, out_grad):
        grad = self.actfunc_grad(self.unnormalized_outputs) * out_grad  # (dOut / dJ) * (dJ / wx+b (unnormalized outputs))

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
    def __init__(self, layer_dims: list[int], actfunc_list: list[str]):
        self.layers : list[Layer] = []

        for i in range(1, len(layer_dims)):
            curr_layer = Layer(layer_dims[i - 1], layer_dims[i], actfunc_list[i - 1])
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

    def grad_descent(self, batch_size=1):
        for layer in self.layers:
            for neuron in layer.neurons:
                neuron.w -= 0.001 * (neuron.grad_w / batch_size)
                neuron.b -= 0.001 * (neuron.grad_b / batch_size)

"""
LOSS FUNCTIONS
"""

def MSEError(pred: np.array, true: np.array):
    loss_grad = pred - true
    loss = 0.5 * np.sum((pred - true) ** 2)

    return loss_grad, loss

# FIX THIS LATER
def LogLossError(pred: np.array, true: np.ndarray):
    # 1 because the gradient is ingested with the final softmax layer gradient
    loss_grad = 1 

    loss = 0
    for i in range(len(true)):
        loss += -np.log(pred[i] + 0.001) if true[i]==1.0 else 0.0

    return loss_grad, loss


"""
TRAINING MODEL
"""


traindata = 'neural-net-scratch/train-images-idx3-ubyte'
labeldata = 'neural-net-scratch/train-labels-idx1-ubyte'

imagearr = idx2numpy.convert_from_file(traindata)
imagearr_flattened = imagearr.reshape(imagearr.shape[0], -1).astype(np.float32) / 255.0

labelarr = idx2numpy.convert_from_file(labeldata)

actfunc_list = ["relu", "relu", "softmax"]
mlp = MLP([784, 128, 64, 10], actfunc_list)


# loop
# for epoch in range(50):

batch_size = 32
running_loss = 0.0
for epoch in range(10):
    running_loss = 0.0
    for i in range(0, len(imagearr), batch_size):
        batch_loss = 0.0
        in_batch = imagearr_flattened[i:i+batch_size]
        out_batch = labelarr[i:i+batch_size]

        for in_data, out_data in zip(in_batch, out_batch):
            y_true = np.zeros(10)
            y_true[int(out_data)] = 1.0

            out = mlp.forward(in_data)
            loss_grad, loss = LogLossError(out, y_true)
            batch_loss += loss
            mlp.backward(loss_grad)

        batch_loss /= batch_size
        mlp.grad_descent(len(in_batch))
        # upd zero grad
        mlp.zero_grad()

        running_loss += batch_loss

        # print(f"Batch {i/32}/{math.ceil(len(imagearr)/batch_size)} complete")

    running_loss /= math.ceil(len(imagearr)/batch_size) # number of batches

    print(f"Epoch {epoch+1}: {running_loss}")


"""
TESTING MODEL
"""

testdata = 'neural-net-scratch/t10k-images-idx3-ubyte'
testlabels = 'neural-net-scratch/t10k-labels-idx1-ubyte'


testarr = idx2numpy.convert_from_file(testdata)
testarr_flattened = testarr.reshape(testarr.shape[0], -1).astype(np.float32) / 255.0

testlabels = idx2numpy.convert_from_file(testlabels)

correct_cnt = 0

for img, label in zip(testarr_flattened, testlabels):
    pred_probs = mlp.forward(img)
    pred_class = np.argmax(pred_probs)
    true_class = int(label)

    if (pred_class == true_class):
        correct_cnt += 1

accuracy = correct_cnt / len(testlabels)
print(correct_cnt, len(testlabels))
print(accuracy)
