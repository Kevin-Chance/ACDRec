import math
import torch

@torch.no_grad()
def uot_weights(cost, tau_row=1.0, tau_col=0.1, eta=0.01, iters=30):
    if cost.ndim != 2 or cost.numel() == 0:
        raise ValueError('UOT expects a nonempty [batch, codes] cost')
    if min(tau_row, tau_col) < 0 or eta <= 0 or iters < 1:
        raise ValueError('UOT requires nonnegative tau, positive eta and iterations')
    d = cost.detach().double().clamp_min(0)
    if not torch.isfinite(d).all():
        raise FloatingPointError('Non-finite UOT cost')
    b, k = d.shape
    log_p, log_q = (-math.log(b), -math.log(k))
    log_g = torch.full_like(d, log_p + log_q)
    for _ in range(iters):
        row = torch.logsumexp(log_g, dim=1, keepdim=True) - log_p
        col = torch.logsumexp(log_g, dim=0, keepdim=True) - log_q
        log_g = log_g - eta * (d + tau_row * row + tau_col * col)
    if not torch.isfinite(log_g).all():
        raise FloatingPointError('UOT diverged; reduce eta or inspect cost scale')
    weights = torch.softmax(log_g, dim=1).to(cost.dtype)
    return weights
