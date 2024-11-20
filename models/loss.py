import torch.nn as nn
from numpy.matrixlib.defmatrix import matrix
from models.gram-matrix import GramMatrix

class ContentLoss(nn.Module):
    def __init__(self, target):
        super(ContentLoss, self).__init__()
        self.target = target.detach()

    def forward(self, input):
        self.loss = nn.functional.mse_loss(input, self.target)
        return input

class StyleLoss(nn.Module):
    def __init__(self, target):
        super(StyleLoss, self).__init__()
        self.target = GramMatrix()(target).detach()

    def forward(self, input):
        gram_input = GramMatrix()(input)
        self.loss = nn.functional.mse_loss(gram_input, self.target)
        return input
