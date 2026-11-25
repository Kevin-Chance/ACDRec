import torch
from torch import nn
from .losses import squared_l2_loss

class VectorQuantizer(nn.Module):

    def __init__(self, n_e, e_dim, beta=0.25):
        super().__init__()
        self.n_e = n_e
        self.e_dim = e_dim
        self.beta = beta
        self.embedding = nn.Embedding(self.n_e, self.e_dim)
        self.embedding.weight.data.uniform_(-1.0 / self.n_e, 1.0 / self.n_e)

    def forward(self, x):
        latent = x.view(-1, self.e_dim)
        embeddings_weight = self.embedding.weight
        d = torch.sum(latent ** 2, dim=1, keepdim=True) + torch.sum(embeddings_weight ** 2, dim=1, keepdim=True).t() - 2 * torch.matmul(latent, embeddings_weight.t())
        indices = torch.argmin(d, dim=-1)
        x_q = self.embedding(indices).view(x.shape)
        commitment_loss = squared_l2_loss(x_q.detach(), x)
        codebook_loss = squared_l2_loss(x_q, x.detach())
        loss = codebook_loss + self.beta * commitment_loss
        x_q = x + (x_q - x).detach()
        indices = indices.view(x.shape[:-1])
        return (x_q, loss, indices, d)
