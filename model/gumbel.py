import torch

def gumbel_softmax(logits, tau=1, hard=False, eps=1e-10, dim=-1):
    gumbels = (
        -torch.empty_like(logits, memory_format=torch.legacy_contiguous_format).exponential_().log()
    )  # ~Gumbel(0,1)
    gumbels = (logits + gumbels) / tau  # ~Gumbel(logits,tau)
    y_soft = gumbels.softmax(dim)

    if hard:

        index = torch.multinomial(y_soft, 1)
        y_hard = torch.zeros_like(logits, memory_format=torch.legacy_contiguous_format).scatter_(0, index, 1.0)
        ret = (y_hard - y_soft).detach() + y_soft
    else:
        # Reparametrization trick.
        ret = y_soft
    return ret