import gym
import math
import random
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from collections import namedtuple, deque
from itertools import count
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T

a = torch.arange(10).reshape(5,2)
b = torch.arange(15).reshape(5,3)
print(a)
for i, j in zip(torch.split(a,1), torch.split(b,1)):
    print(i)
    print(j)