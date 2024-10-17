import random

import numpy as np
import torch
from torch import nn
from torchsummary import summary
import torch.nn.functional as F
from torchvision import models
import torch.nn.init as init
# seed = 1
# torch.manual_seed(seed)
# torch.cuda.manual_seed(seed)
# np.random.seed(seed)
# random.seed(seed)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Tudui(nn.Module):
    def __init__(self,L=201):
        super(Tudui, self).__init__()
        self.model1 = nn.Sequential(
            nn.Conv2d(1, 40,kernel_size=(2,2)),
            nn.ReLU(),
            nn.Conv2d(40, 1, kernel_size=(1, 1)),
            # nn.LeakyReLU(),
            nn.Flatten(),
            nn.Linear(L - 1,1),
            # nn.ReLU(),
            # nn.Linear(2, 1)

        )
        self.conv_output = None

    def forward(self, x):
        x=self.model1(x)
        return x
if __name__ == '__main__':
    tudui = Tudui()
    tudui=tudui.to(device)
    a = summary(tudui, input_size=(1,201,2))
    print(a)
    # params = list(tudui.parameters())
    # k = 0
    # for i in params:
    #     l = 1
    #     print("该层的结构：" + str(list(i.size())))
    #     for j in i.size():
    #         l *= j
    #     print("该层参数和：" + str(l))
    #     k = k + l
    # print("总参数数量和：" + str(k))
