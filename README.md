# Unlearning a quantized time-series model (proof of concept)
Does time-series unlearning still work after the model has been quantized to integers? See `docs/DESIGN.md` for the setup and the rules fixed in advance, and `results/RESULTS_TABLES.md` for the tables.

```
python steps/01_download.py
python steps/02_preprocess.py
python steps/03_train.py --arch dlinear        # also mlp, patchtst
python steps/04_quantize_unlearn.py --arch dlinear --seed 0
python steps/05_report.py
```
