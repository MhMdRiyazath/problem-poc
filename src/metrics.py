import numpy as np, torch
from sklearn.metrics import roc_auc_score
import config as C

@torch.no_grad()
def win_loss(m, x, y):
    m.eval(); return torch.cat([((m(x[i:i + 1024]) - y[i:i + 1024]) ** 2).mean(1) for i in range(0, len(x), 1024)])
def mse(m, x, y): return win_loss(m, x, y).mean().item()

def reference(start, retrained, D):
    """start = model before unlearning, retrained = model trained without client 10; both in the SAME precision as the model under test."""
    fg, cg = win_loss(retrained, D["Xf_e"], D["Yf_e"]), win_loss(retrained, D["Xc"], D["Yc"]); sf = mse(start, D["Xf_e"], D["Yf_e"])
    return dict(start_f=sf, gold_f=fg.mean().item(), start_retain=mse(start, D["Xrt"], D["Yrt"]), gap=(fg.mean().item() - sf) / fg.mean().item(), fg=fg, cg=cg)

def evaluate(m, ref, D):
    """forget_score: 0 = still remembers client 10 (like the start model), 1 = as ignorant as the retrained model.
    mia_auc: attacker scores windows by (retrained loss - model loss); client-10 windows vs never-seen clients; 0.5 = cannot tell them apart.
    retain_ratio: retain-client test error divided by the start model's."""
    f, c = win_loss(m, D["Xf_e"], D["Yf_e"]), win_loss(m, D["Xc"], D["Yc"])
    score = (f.mean().item() - ref["start_f"]) / (ref["gold_f"] - ref["start_f"])
    s = torch.cat([ref["fg"] - f, ref["cg"] - c]).numpy(); lab = np.r_[np.ones(len(f)), np.zeros(len(c))]
    auc = roc_auc_score(lab, s) if np.ptp(s) > 0 else 0.5; ratio = mse(m, D["Xrt"], D["Yrt"]) / ref["start_retain"]
    return dict(forget_score=score, mia_auc=auc, retain_ratio=ratio, passed=bool(C.FORGOTTEN_RANGE[0] <= score <= C.FORGOTTEN_RANGE[1] and ratio <= C.MAX_RETAIN_DAMAGE))
