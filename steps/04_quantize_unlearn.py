"""Train (done) -> quantize -> unlearn the quantized model using only its integer weights -> check.
   For each architecture, seed, quantization technique and bit width: quantize the original and the retrained model identically, run every unlearning method once
   (one fixed setting, no tuning), judge the model after the last step against the equally quantized retrained model. Full precision is run the same way as a reference."""
import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import argparse, csv, time, torch, config as C
from src.data import load_processed
from src.models import make_model
from src.quantize import quantize
from src.metrics import reference, evaluate
from src.unlearn import unlearn
torch.set_num_threads(1)
ap = argparse.ArgumentParser(); ap.add_argument("--arch", required=True); ap.add_argument("--seed", type=int, required=True); a = ap.parse_args()
D = load_processed(); t0 = time.time(); out = open(C.RES / f"raw_{a.arch}_seed{a.seed}.csv", "w", newline="")
w = csv.DictWriter(out, fieldnames=["arch", "seed", "tech", "bits", "method", "hp", "changed", "forget_score", "mia_auc", "retain_ratio", "passed", "gap"]); w.writeheader()
def load(name): m = make_model(a.arch); m.load_state_dict(torch.load(C.CKPT / a.arch / f"seed{a.seed}_{name}.pt")); return m.eval()
orig, retrained = load("original"), load("retrained")
conds = [("fp", 32, orig, retrained, False)] + [(t, b, quantize(orig, t, b), quantize(retrained, t, b), True) for t in C.QUANT for b in C.BITS]
for tech, bits, m0, g, integer_only in conds:
    ref = reference(m0, g, D)
    for method, hp in C.METHODS.items():
        m, ch = unlearn(m0, method, hp, D, a.arch, integer_only, a.seed); e = evaluate(m, ref, D)
        w.writerow(dict(arch=a.arch, seed=a.seed, tech=tech, bits=bits, method=method, hp=hp, changed=ch if integer_only else "", gap=ref["gap"], **e)); out.flush()
    print(f"[{time.time()-t0:.0f}s] {a.arch} seed {a.seed} {tech} {bits}-bit done", flush=True)
