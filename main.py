from fastapi import FastAPI
import uvicorn
import base64
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import os
from fastapi.middleware.cors import CORSMiddleware


# --------------------------- MODEL Structure --------------------------------
factors = [1, 1, 1, 1, 1 / 2, 1 / 4, 1 / 8, 1 / 16, 1 / 32]

# PixelNorm : L2 norm on each pixel (dim=1)

class PixelNorm(nn.Module):
  def __init__(self):
    super(PixelNorm, self).__init__()
  def forward(self, x):
    return x / torch.sqrt(torch.mean(x ** 2, dim=1, keepdim=True) + 1e-8)

# Linear (dense) layer with scaling and weight initalizations

class WeightedScaledLinear(nn.Module):

  def __init__(self, in_features, out_features, gain=2):

    super(WeightedScaledLinear, self).__init__()

    self.linear = nn.Linear(in_features, out_features)
    self.scale = (gain / in_features)**0.5
    self.bias = self.linear.bias
    self.linear.bias = None

    # initialize linear layer
    nn.init.normal_(self.linear.weight)
    nn.init.zeros_(self.bias)

  def forward(self, x):

    return self.linear(x * self.scale) + self.bias

# Conv2d layer with scaling and weight initalizations

class WeightedScaledConv2d(nn.Module):

  def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, gain=2,):

    super(WeightedScaledConv2d, self).__init__()
    self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
    self.scale = (gain / (in_channels * (kernel_size ** 2))) ** 0.5
    self.bias = self.conv.bias
    self.conv.bias = None

    # initialize conv layer
    nn.init.normal_(self.conv.weight)
    nn.init.zeros_(self.bias)

  def forward(self, x):
    
    return self.conv(x * self.scale) + self.bias.view(1, self.bias.shape[0], 1, 1)

# Controls how much noise is injected with the weights parameter. Noise are injected every growing resolutions

class InjectNoise(nn.Module):

  def __init__(self, channels):

    super(InjectNoise, self).__init__()
    self.weight = nn.Parameter(torch.zeros(1, channels, 1, 1))

  def forward(self, x):
    
    noise = torch.randn((x.shape[0], 1, x.shape[2], x.shape[3]), device=x.device)
    return x + self.weight * noise

# AdaIN : pixelnorm with affine scaling like the ones in batch norm
class AdaIN(nn.Module):
  def __init__(self, channels, w_dim):
    super().__init__()
    self.instance_norm = nn.InstanceNorm2d(channels)
    self.style_scale = WeightedScaledLinear(w_dim, channels)
    self.style_bias = WeightedScaledLinear(w_dim, channels)

  def forward(self, x, w):
    x = self.instance_norm(x)
    style_scale = self.style_scale(w).unsqueeze(2).unsqueeze(3)
    style_bias = self.style_bias(w).unsqueeze(2).unsqueeze(3)
    return style_scale * x + style_bias

"""## Network"""

# 8 linear layers to transform noise of dimension z_dim into style vector of dimension w_dim
# PixelNorm is basically used interchangably with L2 norm here

class MappingNetwork(nn.Module):

  def __init__(self, z_dim, w_dim):

    super(MappingNetwork, self).__init__()

    self.mapping = nn.Sequential(
        PixelNorm(),
        WeightedScaledLinear(z_dim, w_dim),
        nn.ReLU(),
        WeightedScaledLinear(w_dim, w_dim),
        nn.ReLU(),
        WeightedScaledLinear(w_dim, w_dim),
        nn.ReLU(),
        WeightedScaledLinear(w_dim, w_dim),
        nn.ReLU(),
        WeightedScaledLinear(w_dim, w_dim),
        nn.ReLU(),
        WeightedScaledLinear(w_dim, w_dim),
        nn.ReLU(),
        WeightedScaledLinear(w_dim, w_dim),
        nn.ReLU(),
        WeightedScaledLinear(w_dim, w_dim),
    )

  def forward(self, x):
    return self.mapping(x)

#Noise and style vector injection right after upsampling
class GenBlock(nn.Module):

  def __init__(self, in_channels, out_channels, w_dim):
    super(GenBlock, self).__init__()
    self.conv1 = WeightedScaledConv2d(in_channels, out_channels)
    self.conv2 = WeightedScaledConv2d(out_channels, out_channels)
    self.leaky = nn.LeakyReLU(0.2, inplace=True)
    self.inject_noise1 = InjectNoise(out_channels)
    self.inject_noise2 = InjectNoise(out_channels)
    self.adain1 = AdaIN(out_channels, w_dim)
    self.adain2 = AdaIN(out_channels, w_dim)

  def forward(self, x, w):
    x = self.adain1(self.leaky(self.inject_noise1(self.conv1(x))), w)
    x = self.adain2(self.leaky(self.inject_noise2(self.conv2(x))), w)
    return x

#2 convolutions after noise and vector injection, used in both gen and dis

class ConvBlock(nn.Module):
  def __init__(self, in_channels, out_channels):
    super(ConvBlock, self).__init__()
    self.conv1 = WeightedScaledConv2d(in_channels, out_channels)
    self.conv2 = WeightedScaledConv2d(out_channels, out_channels)
    self.leaky = nn.LeakyReLU(0.2)

  def forward(self, x):
    x = self.leaky(self.conv1(x))
    x = self.leaky(self.conv2(x))
    return x

class StyleGenerator(nn.Module):

  def __init__(self, z_dim, w_dim, in_channels, img_channels=3):

    super(StyleGenerator, self).__init__()

    self.starting_constant = nn.Parameter(torch.ones((1, in_channels, 4, 4)))
    self.mapping = MappingNetwork(z_dim, w_dim)
    self.initial_adain1 = AdaIN(in_channels, w_dim)
    self.initial_adain2 = AdaIN(in_channels, w_dim)
    self.initial_noise1 = InjectNoise(in_channels)
    self.initial_noise2 = InjectNoise(in_channels)
    self.initial_conv = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1)
    self.leaky = nn.LeakyReLU(0.2, inplace=True)

    self.initial_rgb = WeightedScaledConv2d(
        in_channels, img_channels, kernel_size=1, stride=1, padding=0
    )
    self.prog_blocks, self.rgb_layers = (
        nn.ModuleList([]),
        nn.ModuleList([self.initial_rgb]),
    )

    for i in range(len(factors) - 1):  # -1 to prevent index error because of factors[i+1]
        conv_in_c = int(in_channels * factors[i])
        conv_out_c = int(in_channels * factors[i + 1])
        self.prog_blocks.append(GenBlock(conv_in_c, conv_out_c, w_dim))
        self.rgb_layers.append(
            WeightedScaledConv2d(conv_out_c, img_channels, kernel_size=1, stride=1, padding=0)
        )

  def fade_in(self, alpha, upscaled, generated):
    # alpha should be scalar within [0, 1], and upscale.shape == generated.shape
    return torch.tanh(alpha * generated + (1 - alpha) * upscaled)

  def forward(self, noise, alpha, steps):
    w = self.mapping(noise)
    x = self.initial_adain1(self.initial_noise1(self.starting_constant), w)
    x = self.initial_conv(x)
    out = self.initial_adain2(self.leaky(self.initial_noise2(x)), w)

    if steps == 0:
        return self.initial_rgb(x)

    for step in range(steps):
        upscaled = nn.functional.interpolate(out, scale_factor=2, mode="bilinear")
        out = self.prog_blocks[step](upscaled, w)

    # The number of channels in upscale will stay the same, while
    # out which has moved through prog_blocks might change. To ensure
    # we can convert both to rgb we use different rgb_layers
    # (steps-1) and steps for upscaled, out respectively
    final_upscaled = self.rgb_layers[steps - 1](upscaled)
    final_out = self.rgb_layers[steps](out)
    return self.fade_in(alpha, final_upscaled, final_out)

# --------------------------------------------------------------------------------
app = FastAPI()
origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
seed_size = 512

model_path = './model/model.pt'
model = StyleGenerator(seed_size, 512, 512)
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()

@app.get("/")
def read_root():
    return {"routes": "/api"}


@app.get("/api")
def read_item():
    noise = torch.randn(1, 512, device=device, dtype=torch.float)
    alpha = 0.6

    fake_img = model(noise.cpu(), alpha, 5)
    del noise
  
    fake_img = torch.moveaxis(fake_img.cpu().detach()*127.5+127.5,1,-1).numpy().astype('uint8')  
    path = './img/save.png'
    plt.imsave(f'{path}', fake_img[0])
    del fake_img


    with open("./img/save.png", "rb") as f:
      image_binary = f.read()

      image = base64.b64encode(image_binary).decode("utf-8")
      return {'image': image}