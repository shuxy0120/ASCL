import torch

from pytorch_metric_learning.distances import CosineSimilarity
from pytorch_metric_learning.utils import common_functions as c_f
from pytorch_metric_learning.utils import loss_and_miner_utils as lmu
from model.metrics.generic_pair_loss import GenericPairLoss


class MultiSimilarityLoss(GenericPairLoss):
    """
    modified from https://github.com/MalongTech/research-ms-loss/
    Args:
        alpha: The exponential weight for positive pairs
        beta: The exponential weight for negative pairs
        base: The shift in the exponent applied to both positive and negative pairs
    """

    def __init__(self, alpha=2, beta=50, base=0.5, **kwargs):
        super().__init__(mat_based_loss=True, **kwargs)
        self.alpha = alpha
        self.beta = beta
        self.base = base
        self.add_to_recordable_attributes(
            list_of_names=["alpha", "beta", "base"], is_stat=False
        )

    def _compute_loss(self, mat, pos_mask, neg_mask, c):
        # c2 = c
        N = c.size(0)
        # c = c.unsqueeze(-1)
        # c = torch.abs(c.expand(N, N) - c.expand(N, N).T)
        # c = torch.where(c > c.T, c, c.T)
        # labelp = v_mask[i].expand(N, N).eq(v_mask[i].expand(N, N).t()).float()
        # c = c.expand(N, N)*(c.expand(N, N).t())
        # c = c.expand(36, 36)
        # c = torch.where(c < c.T, c, c.T)
        # certainty_max = torch.where(c > c.T, c, c.T)
        # certainty_min = torch.where(c < c.T, c, c.T)
        # certainty = torch.where(pos_mask == 1, certainty_max, certainty_min)
        # typ = qid[i]
        m = c
        pos_exp = self.distance.margin(mat, self.base)
        neg_exp = self.distance.margin(self.base, mat)
        pos_loss = (1.0 / self.alpha) * lmu.logsumexp(
            self.alpha * (pos_exp), keep_mask=pos_mask.bool(), add_one=True
        )
        neg_loss = (1.0 / self.beta) * lmu.logsumexp(
            self.beta * (neg_exp), keep_mask=neg_mask.bool(), add_one=True
        )
        return {
            "loss": {
                "losses": pos_loss + neg_loss,
                "indices": c_f.torch_arange_from_size(mat),
                "reduction_type": "element",
            }
        }

    def get_default_distance(self):
        return CosineSimilarity()
