import torch
import torch.nn as nn
from torch.nn import init
import functools
from torch.optim import lr_scheduler
import random
import numpy as np
import time
from PIL import Image
import torch.utils.data as data
from .Resnet import ResnetGenerator
from .help_function import init_func,patch_instance_norm_state_dict,get_transform,tensor2im
import os

norm_layer = functools.partial(nn.InstanceNorm2d, affine=False, track_running_stats=False)
generator=ResnetGenerator(input_nc=3,output_nc=3,ngf=64, norm_layer=norm_layer, use_dropout=False, n_blocks=9, padding_type='reflect')


state=torch.load(os.getcwd().replace('\\','/')+'/models/model/generator.pth')
if hasattr(state, '_metadata'):
  del state._metadata

# patch InstanceNorm checkpoints prior to 0.4
for key in list(state.keys()):  # need to copy keys here because we mutate in loop
    patch_instance_norm_state_dict(state, generator, key.split('.'))

generator.load_state_dict(state)

device = torch.device("cuda")
#gpu 위에꺼 cpu 아래꺼
#device = torch.device("cpu")
generator.to(device)

transform=get_transform()

def generate_gan_img(raw_img):
    frame = Image.fromarray(np.uint8(raw_img)).convert('RGB')
    #Image.fromarray(np.uint8(frame))
    A = transform(frame)
    real = A.to(device)
    start_time = time.time()
    fake = generator(real.unsqueeze(0))
    img=tensor2im(fake)
    return img

