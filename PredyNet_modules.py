import torch
import torch.nn as nn 
import torch.nn.functional as F 
from torch.autograd import Variable 

import numpy as np 

class PredyNet(torch.nn.Module):
    def __init__(self, inSize, dSize):
        super(PredyNet, self).__init__()

        # Network layers
        self.eLayer = nn.Linear(dSize, inSize, bias=True)
        self.yLayer = nn.Linear(inSize, dSize, bias=True)

    def forward(self, Input, yPrev, ePrev):
        err = self.eLayer(yPrev) - Input
        y = F.tanh(self.yLayer(ePrev))
        Rec = self.eLayer(yPrev)
        
        yPrev = y
        ePrev = err

        return err, ePrev, y, yPrev, Rec

    
