from typing import Optional

import torch
from torch.distributions.categorical import Categorical
from einops import rearrange, reduce
from torch import einsum

class CategoricalMasked(Categorical):
    def __init__(self, logits: torch.Tensor, mask: Optional[torch.Tensor] = None):
        self.mask = mask
        self.reshaped = False
        if mask is None:
            super(CategoricalMasked, self).__init__(logits=logits)
        else:
            self.mask_value = torch.finfo(logits.dtype).min
            logits.masked_fill_(~self.mask, self.mask_value)
            super(CategoricalMasked, self).__init__(logits=logits)

    def entropy(self):
        if self.mask is None:
            return super().entropy()
        p_log_p = self.logits*self.probs
        p_log_p = torch.where(self.mask,p_log_p,torch.tensor(0, dtype=p_log_p.dtype, device=p_log_p.device))
        entropy = -p_log_p.sum(-1)
        return entropy

class CategoricalMap(Categorical):
    def __init__(self, logits: torch.Tensor, mask: Optional[torch.Tensor] = None):
        self.batch, _, self.height, self.width = logits.size()  # Tuple[int]
        logits = rearrange(logits, "b a h w -> (b h w) a")
        if mask is not None:
            mask = rearrange(mask, "b  h w -> b (h w)")
            self.mask = mask.to(dtype=torch.float32)
        else:
            self.mask = torch.ones((self.batch, self.height * self.width), dtype=torch.float32)
        self.nb_agent = reduce(self.mask, "b (h w) -> b", "sum", b=self.batch, h=self.height, w=self.width)
        super(CategoricalMap, self).__init__(logits=logits)

    def sample(self) -> torch.Tensor:
        action_grid = super().sample()
        action_grid = rearrange(action_grid, "(b h w) -> b h w", b=self.batch, h=self.height, w=self.width)
        return action_grid

    def log_prob(self, action: torch.Tensor) -> torch.Tensor:
        action = rearrange(action, "b h w -> (b h w)", b=self.batch, h=self.height, w=self.width)
        log_prob = super().log_prob(action)
        log_prob = rearrange(log_prob, "(b h w) -> b (h w)", b=self.batch, h=self.height, w=self.width)
        # log_probs where there are agents
        log_prob = einsum("ij,ij->ij", log_prob, self.mask)
        # sum log_prob over agents
        log_prob = reduce(log_prob,  "b (h w) -> b", "sum", b=self.batch, h=self.height, w=self.width)
        return log_prob

    def entropy(self) -> torch.Tensor:
        entropy = super().entropy()
        entropy = rearrange(entropy, "(b h w) -> b (h w)", b=self.batch, h=self.height, w=self.width)
        # entropy where there are agents
        entropy = einsum("ij,ij->ij", entropy, self.mask)
        # sum entropy over agents
        entropy = reduce(entropy, "b (h w) -> b", "sum", b=self.batch, h=self.height, w=self.width)
        return entropy / self.nb_agent
