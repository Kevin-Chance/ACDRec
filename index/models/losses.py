from torch.nn import functional as F


def squared_l2_loss(prediction, target):
    """Mean per-item squared L2 norm, summing the final vector dimension.

    This implements the per-item norms in Eq. (12); unlike elementwise MSE,
    it does not divide by the feature or latent dimension.
    """
    return F.mse_loss(prediction, target, reduction='none').sum(dim=-1).mean()
