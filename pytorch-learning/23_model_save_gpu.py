import torch
import torch.nn as nn

# Save on GPU, Load on CPU
device = torch.device("mps")
model.to(device)
torch.save(model.state_dict(), PATH)


device = torch.device("cpu")
model = Model(*args, **kwargs)
model.load_state_dict(torch.load(PATH, map_location=device))


# Save on GPU, Load on GPU
device = torch.device('mps')
model.to(device)
torch.save(model.state_dict(), PATH)

model = Model(*args, **kwargs)
model.load_state_dict(torch.load(PATH))
model.to(device)

# Save on CPU, Load on GPU
torch.save(model.state_dict(), PATH)

device = torch.device('mps')
model = Model(*args, **kwargs)
model.load_state_dict(torch.load(PATH, map_location='mps:0')) # choose whatever gpu 
model.to(device)