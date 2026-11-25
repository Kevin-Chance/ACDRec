import torch.nn as nn
from torch.nn.init import xavier_normal_

class MLPLayers(nn.Module):

    def __init__(self, layers):
        super(MLPLayers, self).__init__()
        self.layers = layers
        mlp_modules = []
        for idx, (input_size, output_size) in enumerate(zip(self.layers[:-1], self.layers[1:])):
            mlp_modules.append(nn.Dropout(p=0.0))
            mlp_modules.append(nn.Linear(input_size, output_size))
            if idx != len(self.layers) - 2:
                mlp_modules.append(nn.ReLU())
        self.mlp_layers = nn.Sequential(*mlp_modules)
        self.apply(self.init_weights)

    def init_weights(self, module):
        if isinstance(module, nn.Linear):
            xavier_normal_(module.weight.data)
            if module.bias is not None:
                module.bias.data.fill_(0.0)

    def forward(self, input_feature):
        return self.mlp_layers(input_feature)
