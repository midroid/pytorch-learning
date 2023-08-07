# https://pytorch.org/tutorials/beginner/basics/saveloadrun_tutorial.html

import torch
import torchvision.models as models

# Saving model
model = models.vgg16(weights='IMAGENET1K_V1')
torch.save(model.state_dict(), 'model_weights.pth')

# Loading model
model = models.vgg16()
model.load_state_dict(torch.load('model_weights.pth'))
model.eval()


# or 
torch.save(model, 'model.pth')
model = torch.load('model.pth')
