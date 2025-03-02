import torch
import os
import matplotlib.pyplot as plt
import numpy as np
from torch import nn
from tqdm import tqdm
from src.config import OPT_PATH

def optimize_lr_schedule(best_params, total_steps=24000, peak_lr=3e-4, min_lr=1e-10,\
                          lr=5e-9, max_steps=10000, warmup=2160, interval=1000, name="400M"):
    """
    Optimize the learning rate schedule using the fitted MPL model.

    Args:
        data (dict): Dataset with steps, lrs, and losses.
        train_set (list): List of training file names.
        best_params (list): Fitted MPL parameters [L0, A, alpha, B, C, beta, gamma].
        total_steps (int): Total steps in the schedule.
        peak_lr (float): Initial peak learning rate.
        min_lr (float): Minimum learning rate threshold.
        lr (float): Learning rate for optimization.
        max_steps (int): Maximum optimization steps.

    Returns:
        np.ndarray: Optimized learning rate schedule.
    """
    # Initialize MPL with fitted parameters (frozen)
    L0, A, alpha, B, C, beta, gamma = best_params

    # Initialize Delta (learnable LR reductions)
    delta = nn.Parameter(torch.zeros(total_steps-warmup, dtype=torch.float64), requires_grad=True)
    warmup_bias = 0.5 * peak_lr * warmup
    optimizer = torch.optim.Adam([delta], lr=lr)
    loss = 0.0

    # Optimization loop
    for opt_step in tqdm(range(max_steps), desc="Optimizing LR Schedule"):
        optimizer.zero_grad()

        # Compute LR schedule from Delta
        eta = peak_lr - torch.cumsum(delta.clamp(min=0), dim=0)  # Ensure non-negative reductions
        eta = torch.clamp(eta, min=min_lr)  # Enforce minimum LR

        lr_sum = torch.cumsum(eta, dim=0) + warmup_bias
        lr_sum = torch.concatenate([torch.tensor([0]), lr_sum], dim=0) 
        LD = torch.sum(delta * (1 - (1 + C * eta ** (-gamma) * (lr_sum[-1] - lr_sum[:-1])) ** (-beta)))
        pred = L0 + A * lr_sum[-1] ** (-alpha) - B * LD
        pred.backward()
        optimizer.step()

        # Enforce constraints: Delta >= 0, sum(Delta) <= peak_lr
        with torch.no_grad():
            delta.clamp_(min=0, max=peak_lr)
            eta = peak_lr - torch.cumsum(delta, dim=0)             
            delta.masked_fill_(eta <= min_lr, 0)
            opt_lr = eta.detach().numpy()
            loss = pred.item()
            if opt_step % interval == 0:
                print(f"Iteration {opt_step}, loss: {loss}")
                print(opt_lr[:5], " ... ", opt_lr[-5:])
                grad_norm = torch.norm(delta.grad).item()
                print(f"Last 5-step gradients: {delta.grad[-5:]}")
                print(f"Gradient norm: {grad_norm}")

    print(f"Final loss: {loss}")
    os.makedirs(OPT_PATH, exist_ok=True)
    np.save(os.path.join(OPT_PATH, f"{name}.npy"), opt_lr)
    plt.figure()
    plt.plot(np.arange(warmup, total_steps), opt_lr)
    plt.grid(True)
    plt.xlabel("Step")
    plt.ylabel("Learning Rate")
    plt.title(f"Optimized Learning Rate Schedule")
    plt.savefig(os.path.join(OPT_PATH, f"{name}.png"))
    plt.close()
    return opt_lr