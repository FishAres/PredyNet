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

class PredyNet_LSTM(torch.nn.Module):
    def __init__(self, inSize, dSize, hSize):
        super(PredyNet_LSTM, self).__init__()
        self.hSize = hSize

        # LSTM gets error and action as input
        lstm_input_size = inSize + dSize
        self.RLayer = nn.LSTM(lstm_input_size, self.hSize)
        self.PredLayer = nn.Linear(self.hSize, inSize)
        self.hidden = self.init_hidden()

    def init_hidden(self):
        return (torch.randn(1, 1, self.hSize),
                torch.randn(1, 1, self.hSize))
    
    def forward(self, Input):
        latent_pred, self.hidden = self.RLayer(Input.view(1, 1, -1), self.hidden)
        pred = self.PredLayer(F.tanh(latent_pred.detach()))

        return pred

class action_LSTM(torch.nn.Module):
    def __init__(self, inSize, dSize, hSize):
        super(action_LSTM, self).__init__()
        self.hSize = hSize

        # LSTM gets error and action as input
        # and an additional term for reward magnitude
        lstm_input_size = inSize + dSize
        self.RLayer = nn.LSTM(lstm_input_size, self.hSize)
        self.PredLayer = nn.Linear(self.hSize, inSize)
        self.actLayer = nn.Linear(self.hSize, dSize)
        self.hidden = self.init_hidden()

    def init_hidden(self):
        return (torch.randn(1, 1, self.hSize),
                torch.randn(1, 1, self.hSize))
    
    def forward(self, Input):
        latent_pred, self.hidden = self.RLayer(Input.view(1, 1, -1), self.hidden)
        pred = self.PredLayer(F.tanh(latent_pred.detach()))
        act_pred = F.softmax(self.actLayer(latent_pred.detach()))
        return pred, act_pred