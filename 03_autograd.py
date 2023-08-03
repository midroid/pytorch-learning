import torch

x = torch.randn(3, requires_grad=True)
print(x)

y = x + 2
print(y)

z = y*y*2
# z = z.mean()
print(z)

v = torch.tensor([0.1, 1.0, 0.001], dtype=torch.float32)

z.backward(v) # dz/dx
print(x.grad)

# 3 ways
# x.requires_grad_(False) #1
# print(x)
x.detach() #2
print(x)
with torch.no_grad(): #3
    ab = x + 2
    print(ab)


weights = torch.ones(4, requires_grad=True)
for epoch in range(3):
    model_output = ( weights * 3 ).sum()
    model_output.backward()
    print(weights.grad)
    weights.grad.zero_()


optimizer = torch.optim.SGD(weights, lr=0.01)
optimizer.step()
optimizer.zero_grad()


weights = torch.ones(5, requires_grad=True)
z.backward()
weights.grad.zero_()





