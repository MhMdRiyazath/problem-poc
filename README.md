# Unlearning a quantized time-series model: proof of concept

**What we are doing.** Showing that there is no way today to make a quantized time-series forecasting model forget one client's data when no full-precision copy is available.

**What we found.** Below INT8, the existing time-series unlearning methods leave the quantized model unchanged, so nothing is forgotten. This holds for 3 models, 3 ways of quantizing, 5 sizes and 7 methods.

- Setup and rules fixed in advance: `docs/DESIGN.md`
- Result and tables: `docs/RESULTS.md`, `results/RESULTS_TABLES.md`
- How this supports the four gaps: `docs/GAPS.md`

```
python steps/01_download.py
python steps/02_preprocess.py
python steps/03_train.py --arch dlinear        # also mlp, patchtst
python steps/04_quantize_unlearn.py --arch dlinear --seed 0
python steps/05_report.py
```
