from torchvision import transforms
from enum import Enum

class Transformer(Enum):
    ALEXNET = transforms.Compose([
        transforms.Resize(256, antialias=None),
        transforms.CenterCrop(224),
        transforms.ToPILImage(),
        transforms.ToTensor()
    ])
