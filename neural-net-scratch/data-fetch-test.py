import idx2numpy
import numpy as np
import matplotlib.pyplot as plt

file = 'neural-net-scratch\\train-images-idx3-ubyte'

arr = idx2numpy.convert_from_file(file)

# plt.imshow(arr[4], cmap=plt.cm.binary)
# plt.show()

arr_flatten = arr.reshape(arr.shape[0], -1)

print(arr_flatten.shape)