import math

import torch
from torch import nn
from torch.nn import init
from torch.nn import functional as F


def norm_prob(mus, logsigmas, values):
    mus = torch.unsqueeze(mus, 2)
    logsigmas = torch.unsqueeze(logsigmas, 2)
    values = torch.unsqueeze(values, 1)
    var = torch.exp(logsigmas)**2
    log_prob =  (-((values - mus) ** 2) / (2 * var)).sum(dim=-1) - logsigmas.sum(dim=-1) - values.shape[-1] * math.log(math.sqrt((2 * math.pi)))
    return torch.exp(log_prob)


class SlotAttentionBase(nn.Module):
    """
    Slot Attention module
    """
    def __init__(self, num_slots, dim, iters=5, eps=1e-8, hidden_dim=128):
        super().__init__()
        self.num_slots = num_slots
        self.iters = iters
        self.eps = eps
        self.scale = dim ** -0.5

        self.slots_mu = nn.Parameter(torch.randn(1, 1, dim))

        self.slots_logsigma = nn.Parameter(torch.zeros(1, 1, dim))
        init.xavier_uniform_(self.slots_logsigma)

        self.to_q = nn.Linear(dim, dim, bias=False)
        self.to_k = nn.Linear(dim, dim, bias=False)
        self.to_v = nn.Linear(dim, dim, bias=False)

        self.gru = nn.GRUCell(dim, dim)

        hidden_dim = max(dim, hidden_dim)

        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.ReLU(inplace = True),
            nn.Linear(hidden_dim, dim)
        )

        self.norm_input  = nn.LayerNorm(dim)
        self.norm_slots  = nn.LayerNorm(dim)
        self.norm_pre_ff = nn.LayerNorm(dim)
        
    def step(self, slots, k, v, b, n, d, device, n_s):
        slots_prev = slots

        slots = self.norm_slots(slots)
        q = self.to_q(slots)

        dots = torch.einsum('bid,bjd->bij', q, k) * self.scale
        attn = dots.softmax(dim=1) + self.eps
        attn = attn / attn.sum(dim=-1, keepdim=True)

        updates = torch.einsum('bjd,bij->bid', v, attn)

        slots = self.gru(
            updates.reshape(-1, d),
            slots_prev.reshape(-1, d)
        )

        slots = slots.reshape(b, -1, d)
        slots = slots + self.mlp(self.norm_pre_ff(slots))
        return slots

    def forward(self, inputs, slots=None, *args, **kwargs):
        b, n, d, device = *inputs.shape, inputs.device
        n_s = self.num_slots
        
        if slots is None:
            mu = self.slots_mu.expand(b, n_s, -1)
            sigma = self.slots_logsigma.exp().expand(b, n_s, -1)

            slots = mu + sigma * torch.randn(mu.shape, device = device)

        inputs = self.norm_input(inputs)        
        k, v = self.to_k(inputs), self.to_v(inputs)

        for _ in range(self.iters):
            slots = self.step(slots, k, v, b, n, d, device, n_s)
        slots = self.step(slots.detach(), k, v, b, n, d, device, n_s)

        return slots