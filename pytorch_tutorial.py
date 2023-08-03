import torch
import numpy

x = torch.empty(2, 2, 3, 3)

print(x)

y = torch.rand(2, 2)
print(y)

z = torch.zeros(10)

print(z)

a = torch.ones(10)
print(a)

b = torch.ones(2, 2, dtype=torch.float64)
print(b)

c = torch.ones(2,2, dtype=torch.float64)
print(c)

d = b + c
print(d)

e = torch.add(d, b)
print(e)

e.add_(d)
print(e)

f = torch.sub(e, c)
print(f)

g = torch.mul(f, e)
print(g)

f = torch.rand(5, 3)
print(f)

print(f[:, 0])
print(f[:, 1])
print(f[1, 1])
print(f[1,1].item())

i = torch.rand(4,4)

j = i.view(16)
print(j)

k = i.view(-1, 8)
print(k)
print(k.size())

a = torch.ones(5)
print(a)

b = a.numpy()
print(b)

print(type(b))

a.add_(1)
print(a)
print(b)

a = numpy.ones(5)
print(a)

b = torch.from_numpy(a)
print(b)
print(type(b))

a += 1
print(a)
print(b)

if torch.cuda.is_available():
    print("cuda available")
else:
    print("cuda not available")


print(torch.device)
print(torch.backends)

# From https://pytorch.org/docs/stable/notes/mps.html
if not torch.backends.mps.is_available():
    if not torch.backends.mps.is_build():
        print("MPS not available because current pytorch install was not "
              "built with MPS enabled.")
    else:
        print("MPS not available because the current MacOS version is not 12.3+ "
              "and/or you don not have an MPS-enabled device on this machine.")
else:
    mps_device = torch.device("mps")
    print(mps_device)
    # Create a Tensor directly on the mps device
    x = torch.ones(5, device=mps_device)
    # Or
    x = torch.ones(5, device="mps")

    # Any operation happends on the GPU
    y = x * 2

    print(y)

    # Move your model to mps just like any other device
    # model = YourFavoriteNet()
    # model.to(mps_device)

    # Now every call runs on the GPU
    # pred = model(x)

GPU_DEVICE = "mps"
if torch.backends.mps.is_available():
    device = torch.device(GPU_DEVICE)
    x = torch.ones(5, device=device)
    y = torch.ones(5)
    y = y.to(device)
    z = x + y
    z = z.to("cpu")
    print(x)
    print(y)
    print(z)


x = torch.ones(5, requires_grad=True)
print(x)

