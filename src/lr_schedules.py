import numpy as np

def cosine_lrs(warmup: int, total: int, peak_lr: float, end_lr: float, const_warmup: bool) -> np.ndarray:
    """Generate cosine learning rate schedule."""
    step = np.arange(total)[warmup:]
    warmup_lrs = np.linspace(0, peak_lr, warmup) if not const_warmup else np.full(warmup, peak_lr)
    cosine_lrs = end_lr + 0.5 * (peak_lr - end_lr) * (1 + np.cos(np.pi * (step - warmup) / (total - warmup)))
    return np.concatenate((warmup_lrs, cosine_lrs))

def const_lrs(warmup: int, total: int, lr: float, const_warmup: bool) -> np.ndarray:
    """Generate constant learning rate schedule."""
    warmup_lrs = np.linspace(0, lr, warmup) if not const_warmup else np.full(warmup, lr)
    return np.concatenate((warmup_lrs, np.full(total - warmup, lr)))

def two_stage_lrs(warmup: int, total: int, lr_a: float, lr_b: float, stage_a: int, const_warmup: bool) -> np.ndarray:
    """Generate two-stage learning rate schedule."""
    warmup_lrs = np.linspace(0, lr_a, warmup) if not const_warmup else np.full(warmup, lr_a)
    stage_a_lrs = np.full(stage_a - warmup, lr_a)
    stage_b_lrs = np.full(total - stage_a, lr_b)
    return np.concatenate((warmup_lrs, stage_a_lrs, stage_b_lrs))

def wsd_lrs(warmup: int, total: int, decay: int, peak_lr: float, end_lr: float, const_warmup: bool) -> np.ndarray:
    """Generate WSD learning rate schedule."""
    step = np.arange(total)[decay:]
    warmup_lrs = np.linspace(0, peak_lr, warmup) if not const_warmup else np.full(warmup, peak_lr)
    decay_lrs = peak_lr ** ((total - step) / (total - decay)) * end_lr ** ((step - decay) / (total - decay))
    return np.concatenate((warmup_lrs, np.full(decay - warmup, peak_lr), decay_lrs))

def wsdld_lrs(warmup: int, total: int, decay: int, peak_lr: float, end_lr: float, const_warmup: bool) -> np.ndarray:
    """Generate WSDLD learning rate schedule."""
    step = np.arange(total)[decay:]
    warmup_lrs = np.linspace(0, peak_lr, warmup) if not const_warmup else np.full(warmup, peak_lr)
    decay_lrs = peak_lr * (1 - (step - decay) / (total - decay)) + end_lr * (step - decay) / (total - decay)
    return np.concatenate((warmup_lrs, np.full(decay - warmup, peak_lr), decay_lrs))