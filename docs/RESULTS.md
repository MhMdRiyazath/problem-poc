# Results

Setup in `DESIGN.md`. Pipeline: train in full precision, quantize after training (3 techniques x INT8/6/4/3/2), unlearn the quantized model using only its integer weights, check. Seven time-series unlearning methods, each run once with one fixed setting, no tuning. DLinear 3 seeds, MLP 2 seeds, PatchTST 1 seed.
Full tables: `../results/RESULTS_TABLES.md`. Per-run data: `../results/raw_*.csv`. Figures: `../figures/<model>.png`.

## What the data shows
1. **Below INT8, the unlearning methods leave the integer weights unchanged.** At INT3 and INT2 all 7 methods leave every integer weight unchanged (0% changed), for all three quantization techniques and all three models. At INT4 and INT6, 6 of 7 methods leave the weights unchanged (INT4 per-tensor: 7 of 7). The cause is that a standard-size update is smaller than half a quantization step and is rounded back every step.
2. **The one exception is Gaussian Amnesiac at INT6 and INT4.** Its integer weights do change, but only because the method adds noise to the weights itself; the repair steps that follow cannot move them. Its forget score stays near 0 (0.00 to 0.13), so it moved without forgetting.
3. **INT8 is different:** 6 of 7 methods change 55 to 96% of the integer weights. MaxLoss stays at 0% because its fixed learning rate (1e-4) is below half a bin.
4. **In full precision, none of the 7 methods meets the strict forgetting rule with these fixed settings** (forget score 0.8 to 1.2 and retain error at most 10% worse). They overshoot (RandLabel, ZeroLabel, MaxLoss, Blindspot), undershoot (NoisyLabel, TSU, Gaussian Amnesiac) or damage retain data. So moving at INT8 does not mean unlearning correctly.

## Table: methods stuck, per quantization technique and bit width

**dlinear** (methods stuck out of 7; stuck = fewer than 1% of the integer weights changed)

| Quantization technique | INT8 | INT6 | INT4 | INT3 | INT2 |
|---|---|---|---|---|---|
| per-channel, abs-max | 1/7 | 6/7 | 6/7 | 7/7 | 7/7 |
| per-tensor, abs-max | 1/7 | 6/7 | 7/7 | 7/7 | 7/7 |
| per-channel, 99th-percentile clipping | 1/7 | 6/7 | 6/7 | 7/7 | 7/7 |

**mlp** (methods stuck out of 7; stuck = fewer than 1% of the integer weights changed)

| Quantization technique | INT8 | INT6 | INT4 | INT3 | INT2 |
|---|---|---|---|---|---|
| per-channel, abs-max | 1/7 | 6/7 | 6/7 | 7/7 | 7/7 |
| per-tensor, abs-max | 1/7 | 6/7 | 7/7 | 7/7 | 7/7 |
| per-channel, 99th-percentile clipping | 1/7 | 6/7 | 6/7 | 7/7 | 7/7 |

**patchtst** (methods stuck out of 7; stuck = fewer than 1% of the integer weights changed)

| Quantization technique | INT8 | INT6 | INT4 | INT3 | INT2 |
|---|---|---|---|---|---|
| per-channel, abs-max | 1/7 | 6/7 | 6/7 | 7/7 | 7/7 |
| per-tensor, abs-max | 1/7 | 6/7 | 6/7 | 7/7 | 7/7 |
| per-channel, 99th-percentile clipping | 1/7 | 6/7 | 6/7 | 7/7 | 7/7 |

## What this does and does not show
- It shows: the time-series unlearning methods, used with fixed standard settings on a model quantized to INT6 or below, do not change the integer weights and so cannot unlearn. This holds across three models and three quantization techniques.
- It does not show: that no setting can work. A different learning rate could produce bigger steps. The methods are my simplified re-implementations. The fixed settings are my choice.
- It does not show that INT8 unlearning is correct: the methods move the weights but do not reach the forgetting target in full precision either.
- One dataset, one forgotten client, simulated integer weights, 1 to 3 seeds.
