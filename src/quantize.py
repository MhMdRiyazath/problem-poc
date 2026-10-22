import copy, torch
from .models import qlinears

def quantize(m, tech, bits):
    """Post-training quantization of every Linear weight to signed integers (round to nearest). Biases, LayerNorm and position tables stay full precision.
    tech: 'channel' one scale per output channel (abs-max) | 'tensor' one scale per layer (abs-max) | 'clip' per channel, scale from the 99th percentile (outliers clipped).
    Weights are stored as floats that sit exactly on the integer grid (simulated integer arithmetic); the scale stays fixed afterwards."""
    _, per_tensor, pct = __import__("config").QUANT[tech]; m = copy.deepcopy(m); qmax = 2 ** (bits - 1) - 1
    for l in qlinears(m):
        a = l.weight.detach().abs()
        if per_tensor: top = a.max().reshape(1, 1)
        else: top = a.amax(1, keepdim=True) if pct is None else torch.quantile(a, pct, dim=1, keepdim=True)
        l.qmax = qmax; l.scale = (top / qmax).clamp_min(1e-8); l.weight.data = torch.clamp(torch.round(l.weight.data / l.scale), -qmax, qmax) * l.scale
    return m

def project(m):
    """Snap every weight back onto its integer grid (scale and range fixed)."""
    with torch.no_grad():
        for l in qlinears(m): l.weight.data = torch.clamp(torch.round(l.weight.data / l.scale), -l.qmax, l.qmax) * l.scale
