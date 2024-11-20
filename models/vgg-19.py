import torch.nn as nn
import torchvision.models as models

class VGG19(nn.Module):
    def __init__(self):
        super(VGG19, self).__init__()
        vgg = models.vgg19(pretrained=True).features
        self.features = nn.Sequential(*[vgg[i] for i in range(len(vgg))])

    def forward(self, x):
        outputs = []
        for layer in self.features:
            x = layer(x)
            outputs.append(x)
        return outputs
