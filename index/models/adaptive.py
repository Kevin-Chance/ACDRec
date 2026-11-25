import torch
from torch import nn
from torch.nn import functional as F
from .layers import MLPLayers
from .vq import VectorQuantizer
from .transforms import NonUniformTransform
from .transport import uot_weights
from .losses import squared_l2_loss

class AdaptiveResidualQuantizer(nn.Module):

    def __init__(self, sizes, e_dim, beta, tau_item, tau_code, eta, iters):
        super().__init__()
        self.vq_layers = nn.ModuleList([VectorQuantizer(size, e_dim, beta) for size in sizes])
        self.uot_config = dict(tau_row=tau_item, tau_col=tau_code, eta=eta, iters=iters)

    def forward(self, x, use_sk=True):
        residual, total = (x, 0)
        losses, indices, distances = ([], [], [])
        for level, vq in enumerate(self.vq_layers):
            last = level == len(self.vq_layers) - 1
            custom = self.training and use_sk and last
            if custom:
                latent = residual.reshape(-1, vq.e_dim)
                weight = vq.embedding.weight
                d = (latent.square().sum(1, keepdim=True) + weight.square().sum(1)[None] - 2 * latent @ weight.T).clamp_min(0)
                p = uot_weights(d, **self.uot_config)
                idx = p.argmax(-1)
                hard = F.embedding(idx, weight).view_as(residual)
                loss = squared_l2_loss(hard, residual.detach()) + vq.beta * squared_l2_loss(hard.detach(), residual)
                soft = (p @ weight).view_as(residual)
                q = residual + (soft - residual).detach()
                idx = idx.view(residual.shape[:-1])
            else:
                q, loss, idx, d = vq(residual)
            residual = residual - q
            total = total + q
            losses.append(loss)
            indices.append(idx)
            distances.append(d)
        max_k = max((d.shape[-1] for d in distances))
        padded = [F.pad(d, (0, max_k - d.shape[-1]), value=float('inf')) for d in distances]
        # Eq. (12) sums layer losses; each layer already averages over items.
        return (total, torch.stack(losses).sum(), torch.stack(indices, -1), torch.stack(padded, 1))

class ACDRecTokenizer(nn.Module):

    def __init__(self, in_dim, num_emb_list=(256, 256, 256, 256), e_dim=32, layers=(2048, 1024, 512, 256, 128, 64), beta=0.25, kmeans_iters=100, tau_item=1.0, tau_code=0.1, eta=0.1, iters=30):
        super().__init__()
        sizes = list(num_emb_list)
        if in_dim < 1 or e_dim < 1 or (not 1 <= len(sizes) <= 8) or (min(sizes) < 2):
            raise ValueError('Invalid input, latent or codebook dimensions')
        self.in_dim = in_dim
        self.num_emb_list = sizes
        self.e_dim = e_dim
        self.encoder = MLPLayers([in_dim] + list(layers) + [e_dim])
        self.rq = AdaptiveResidualQuantizer(sizes, e_dim, beta, tau_item, tau_code, eta, iters)
        self.decoder = MLPLayers(([in_dim] + list(layers) + [e_dim])[::-1])
        self.transform = NonUniformTransform(e_dim)
        self.model_config = dict(in_dim=in_dim, num_emb_list=sizes, e_dim=e_dim, layers=list(layers), beta=beta, kmeans_iters=kmeans_iters, tau_item=tau_item, tau_code=tau_code, eta=eta, iters=iters)

    def forward(self, x, use_sk=True):
        h = self.encoder(x)
        d = self.transform(h)
        q, rq_loss, indices, distances = self.rq(d, use_sk=use_sk)
        restored = self.transform.inverse(q)
        out = self.decoder(restored)
        result = (out, rq_loss, indices, distances)
        return result

    @torch.no_grad()
    def get_indices(self, xs, use_sk=False):
        if self.training:
            raise RuntimeError('Call model.eval() before generating persistent indices')
        if use_sk:
            raise ValueError('Persistent raw indices only support deterministic argmin')
        d = self.transform(self.encoder(xs))
        _, _, idx, distances = self.rq(d, use_sk=False)
        return (idx, distances)

    @torch.no_grad()
    def initialize_codebooks(self, x, seed=42):
        from sklearn.cluster import KMeans
        was_training = self.training
        self.eval()
        residual = self.transform(self.encoder(x))
        for level, vq in enumerate(self.rq.vq_layers):
            if len(residual) < vq.n_e:
                raise ValueError(f'Need at least {vq.n_e} initialization items')
            km = KMeans(n_clusters=vq.n_e, n_init=10, max_iter=self.model_config['kmeans_iters'], random_state=seed + level)
            centers = torch.as_tensor(km.fit(residual.cpu().float().numpy()).cluster_centers_, device=residual.device, dtype=residual.dtype)
            vq.embedding.weight.copy_(centers)
            d = torch.cdist(residual, vq.embedding.weight).square()
            residual = residual - vq.embedding(d.argmin(-1))
        self.train(was_training)

    def compute_loss(self, out, quant_loss, xs):
        recon = squared_l2_loss(out, xs)
        return (recon + quant_loss, recon)
