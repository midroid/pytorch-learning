# https://pytorch.org/tutorials/beginner/basics/tensorqs_tutorial.html

import torch
import numpy as np

data = [[1,2], [3,4]]
x_data = torch.tensor(data)

print(x_data)

np_array = np.array(data)
print(np_array)

x_np = torch.from_numpy(np_array)
print(x_np)

x_ones = torch.ones_like(x_data)
print(x_ones)

x_rand = torch.rand_like(x_data, dtype=torch.float)
print(x_rand)

shape = (2,3,)
rand_tensor = torch.rand(shape)
zero_tensor = torch.zeros(shape)
ones_tensor = torch.ones(shape)

print(rand_tensor)
print(zero_tensor)
print(ones_tensor)

tensor = torch.rand(3, 4)

print(f"Shape: {tensor.shape}")
print(f"Datatype: {tensor.dtype}")
print(f"Device : {tensor.device}")

if torch.backends.mps.is_available():
    tensor = tensor.to("mps")


print(f"device: {tensor.device}")

tensor = torch.ones(4, 4)
print(f"first row: {tensor[0]}")
print(f"first column: {tensor[:, 0]}")
print(f"last column: {tensor[:, -1]}")
tensor[:, 1] = 0
print(tensor)

t1 = torch.cat([tensor, tensor, tensor], dim=1)
print(t1)

# Arithmatic operations
# This computes the matrix multiplication between two tensors. y1, y2, y3 will have the same value
# ``tensor.T`` returns the transpose of a tensor
y1 = tensor @ tensor.T
y2 = tensor.matmul(tensor.T)

y3 = torch.rand_like(y1)
torch.matmul(tensor, tensor.T, out=y3)
print(y1)
print(y2)
print(y3)

# This computes the element-wise product. z1, z2, z3 will have the same value
z1 = tensor * tensor
z2 = tensor.mul(tensor)

z3 = torch.rand_like(tensor)
torch.mul(tensor, tensor, out=z3)
print(z1)
print(z2)
print(z3)

agg = tensor.sum()
agg_item = agg.item()
print(agg_item, type(agg_item))

# In Place
print(f"{tensor}")
tensor.add_(5)
print(f"{tensor}")


# Bridge with numpy
t = torch.ones(5)
print(f"t: {t}")
n = t.numpy()
print(f"n: {n}")

t.add_(1)
print(f"t: {t}")
print(f"n: {n}")

n = np.ones(5)
t = torch.from_numpy(n)

print(f"t: {t}")
print(f"n: {n}")

np.add(n, 1, out=n)
print(f"t: {t}")
print(f"n: {n}")


