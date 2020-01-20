import rpi
import torch.nn as nn
import torch


c0 = nn.Conv2d(3, 3, 3, 1, padding=1)
c1 = nn.Conv2d(3, 5, 5, 1, padding=2)
c2 = nn.Conv2d(5, 2, 3, 1, padding=1)

c1_ = nn.Conv2d(3, 5, 3, 1, padding=1)

lin = nn.Linear(3*32*2*32, 4)
last = nn.Softmax()

m = nn.ModuleList([c0, c1, c2, lin, last])

ra = rpi.rolling_arc(m)

x = torch.randn(1,3,32,32)

o1 = ra(x)

ra.replace_conv_n_with(1, c1_)

o2 = ra(x)

