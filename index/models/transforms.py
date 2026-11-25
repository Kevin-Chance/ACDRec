import torch
from torch import nn
from torch.nn import functional as F

class NonUniformTransform(nn.Module):

    def __init__(self, dim, eps=1e-06):
        super().__init__()
        self.eps = float(eps)
        self.a_raw = nn.Parameter(torch.zeros(dim))
        self.b_raw = nn.Parameter(torch.zeros(dim))

    def parameters_ab(self):
        return (F.softplus(self.a_raw.double()) + self.eps, F.softplus(self.b_raw.double()) + self.eps)

    def forward(self, h):
        x = torch.sigmoid(h.double()).clamp(self.eps, 1 - self.eps)
        a, b = self.parameters_ab()
        xa = x.pow(a).clamp(max=1 - self.eps)
        x = -torch.expm1(b * torch.log1p(-xa))
        return x.clamp(self.eps, 1 - self.eps).to(h.dtype)

    def inverse(self, y):
        x = y.double().clamp(self.eps, 1 - self.eps)
        a, b = self.parameters_ab()
        inner = (-torch.expm1(torch.log1p(-x) / b)).clamp(self.eps, 1 - self.eps)
        x = inner.pow(1 / a).clamp(self.eps, 1 - self.eps)
        return (torch.log(x) - torch.log1p(-x)).to(y.dtype)

