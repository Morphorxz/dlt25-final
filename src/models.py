import torch
from torch import nn
from .utils import torch_huber

class MPL(nn.Module):
    """Multi-Power Law model for loss prediction."""
    def __init__(self, L0: float, A: float, alpha: float, B: float, C: float, beta: float, gamma: float):
        super().__init__()
        self.L0 = nn.Parameter(torch.tensor(L0, dtype=torch.float64))
        self.A = nn.Parameter(torch.tensor(A, dtype=torch.float64))
        self.alpha = nn.Parameter(torch.tensor(alpha, dtype=torch.float64))
        self.B = nn.Parameter(torch.tensor(B, dtype=torch.float64))
        self.C = nn.Parameter(torch.tensor(C, dtype=torch.float64))
        self.beta = nn.Parameter(torch.tensor(beta, dtype=torch.float64))
        self.gamma = nn.Parameter(torch.tensor(gamma, dtype=torch.float64))

    def forward(self, S1, lrs, lr_sum, step, lr_gap, loss):
        LD = torch.zeros_like(step, dtype=torch.float64)
        for i, s in enumerate(step):
            LD[i] = torch.sum(lr_gap[1:s+1] * (1 - (1 + self.C * lrs[1:s+1] ** (-self.gamma) * (lr_sum[s] - lr_sum[:s])) ** (-self.beta)))
        pred = self.L0 + self.A * S1 ** (-self.alpha) + self.B * LD
        r = torch.log(loss) - torch.log(pred)
        return torch_huber(0.001, r).sum()

@DeprecationWarning
class MultiPower(nn.Module):
    """Alternative Multi-Power model for loss prediction."""
    def __init__(self, A=0.4, B=200, C=0.25, alpha=0.5, beta=0.15, gamma=0.10, L0=5.0):
        super().__init__()
        self.A = nn.Parameter(torch.tensor(A, dtype=torch.float32))
        self.B = nn.Parameter(torch.tensor(B, dtype=torch.float32))
        self.C = nn.Parameter(torch.tensor(C, dtype=torch.float32))
        self.alpha = nn.Parameter(torch.tensor(alpha, dtype=torch.float32))
        self.beta = nn.Parameter(torch.tensor(beta, dtype=torch.float32))
        self.gamma = nn.Parameter(torch.tensor(gamma, dtype=torch.float32))
        self.L0 = nn.Parameter(torch.tensor(L0, dtype=torch.float32))

    def forward(self, step, eta):
        n = len(eta)
        diff_lr = eta[:-1] - eta[1:]
        _extended_step = torch.cat([torch.tensor([step[0]]), step, torch.tensor([step[-1]])])
        _step_diff = _extended_step[2:] - _extended_step[:-2]
        partial_sum = torch.cumsum(0.5 * eta * _step_diff, dim=0) + 0.5 * eta[0] * step[0]
        fragment_sum = partial_sum[None, 1:] - partial_sum[:-1, None]
        fragment_sum = torch.triu(fragment_sum, diagonal=0)
        x_power = eta[None, 1:] ** (-self.gamma) * fragment_sum
        power = 1 - (1 + self.C * x_power) ** (-self.beta)
        loss_drop = self.B * torch.matmul(diff_lr.unsqueeze(0), power).squeeze(0)
        loss_drop = torch.cat([torch.tensor([0]), loss_drop])
        const_term = self.A * partial_sum ** (-self.alpha) + self.L0
        return const_term - loss_drop