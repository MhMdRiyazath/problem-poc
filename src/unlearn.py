import copy, torch, torch.nn.functional as F
import config as C
from .models import qlinears, make_model
from .quantize import project

def unlearn(m0, method, hp, D, arch, quantized, seed):
    """Time-series unlearning methods, client 10 = forget set, the other 9 clients = retain set. hp = learning rate (noise level for GaussAmnesiac). Adam, 400 steps, batch 64.
      RandLabel      forget windows trained toward random targets, plus the retain loss                      (TS-Unlearn family of label-based baselines)
      ZeroLabel      forget windows trained toward zeros, plus the retain loss                               (same family)
      NoisyLabel     forget windows trained toward noisy copies of their targets, plus the retain loss      (TS-Unlearn baseline)
      MaxLoss        gradient ascent on the forget windows (capped), plus the retain loss
      TSU            TS-Unlearn dual objective: 0.5 x retain loss + 0.5 x loss toward noisy forget targets   (fixed Pareto weight 0.5)
      GaussAmnesiac  Gaussian noise added to the weights (relative std = hp), then 200 retain-only repair steps
      Blindspot      distillation: stay close to the original model on retain windows, copy an untrained network on forget windows
    quantized=True : the model is the deployed quantized one and no full-precision copy exists. Only its integer Linear weights may change; after every step they are snapped back onto the integer grid (scales fixed); biases / norms frozen.
    quantized=False: full-precision model (reference only), all parameters train."""
    m = copy.deepcopy(m0); torch.manual_seed(seed + 7); lr = 3e-4 if method == "GaussAmnesiac" else hp
    params = [l.weight for l in qlinears(m)] if quantized else list(m.parameters())
    for p in m.parameters(): p.requires_grad_(any(p is q for q in params))
    teacher = None
    if method == "Blindspot": torch.manual_seed(1000 + seed); teacher = make_model(arch).eval(); torch.manual_seed(seed + 7)
    steps = C.STEPS
    if method == "GaussAmnesiac":
        with torch.no_grad():
            for p in params: p.add_(torch.randn_like(p) * hp * p.std())
            if quantized: project(m)
        steps = 200
    opt = torch.optim.Adam(params, lr); q0 = [l.weight.detach().clone() for l in qlinears(m0)]; orig = copy.deepcopy(m0).eval()
    for st in range(steps):
        m.train(); bi = torch.randint(0, len(D["Xr"]), (C.BATCH_U,)); bf = torch.randint(0, len(D["Xf"]), (C.BATCH_U,))
        xr, yr, xf, yf = D["Xr"][bi], D["Yr"][bi], D["Xf"][bf], D["Yf"][bf]; lret = F.mse_loss(m(xr), yr)
        noisy = yf + torch.randn_like(yf) * yf.std(1, keepdim=True)
        if method == "GaussAmnesiac": loss = lret
        elif method == "RandLabel": loss = lret + F.mse_loss(m(xf), torch.randn_like(yf))
        elif method == "ZeroLabel": loss = lret + F.mse_loss(m(xf), torch.zeros_like(yf))
        elif method == "NoisyLabel": loss = lret + F.mse_loss(m(xf), noisy)
        elif method == "TSU": loss = 0.5 * lret + 0.5 * F.mse_loss(m(xf), noisy)
        elif method == "MaxLoss": loss = lret - 0.5 * torch.clamp(F.mse_loss(m(xf), yf), max=5.0)
        elif method == "Blindspot":
            with torch.no_grad(): to_r, to_f = orig(xr), teacher(xf)
            loss = F.mse_loss(m(xr), to_r) + lret + F.mse_loss(m(xf), to_f)
        else: raise ValueError(method)
        opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(params, 1.0); opt.step()
        if quantized: project(m)
    changed = sum((a != l.weight.detach()).sum().item() for a, l in zip(q0, qlinears(m))) / sum(a.numel() for a in q0)
    return m.eval(), changed
