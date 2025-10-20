import torch
import torchvision

from torch import nn
def create_vit_model(num_classes: int = 5):
  # create vit pretrained weights, transforms and model
  weights = torchvision.models.ViT_B_16_Weights.DEFAULT
  transforms = weights.transforms()
  model = torchvision.models.vit_b_16(weights=weights)
  # freeze all layers in base model
  for param in model.parameters():
    param.requires_grad = False
  # change the classifier head to suit our problem
  vit.heads = nn.Sequential(nn.Linear(in_features=768,
                                      out_features=5,
                                      bias=True))
  return model, transforms
