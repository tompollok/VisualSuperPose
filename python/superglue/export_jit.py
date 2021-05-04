import torch

from models.superglue import SuperGlue
from models.superpoint import SuperPoint

superpoint = SuperPoint({}).eval()
superglue = SuperGlue({'weights': 'outdoor'}).eval()
torch.jit.save(superpoint, 'SuperPoint.zip')
torch.jit.save(superglue, 'SuperGlue.zip')