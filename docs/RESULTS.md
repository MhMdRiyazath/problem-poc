# Result

## What we tried
Take a time-series forecasting model, quantize it, then ask it to forget one client's data, using the unlearning methods from the time-series literature. No full-precision copy is available and no retraining from scratch is allowed. This is the situation of a model that is already deployed on a device.

3 models (DLinear, MLP, PatchTST) x 3 ways of quantizing x 5 sizes (INT8, 6, 4, 3, 2) x 7 unlearning methods, each run once with one fixed setting. Setup and rules fixed in advance: `DESIGN.md`.

## What we found
**Below INT8, the existing time-series unlearning methods do not change the quantized model at all, so it cannot forget.**

- At INT3 and INT2, all 7 methods change 0% of the model's integer weights, for every way of quantizing and every model.
- At INT4 and INT6, 6 of the 7 methods change 0% of the weights (INT4 per-tensor: 7 of 7 on DLinear and MLP). The seventh, Gaussian Amnesiac, moves weights only through the noise it adds itself, and it does not forget (forget score 0.00 to 0.13).
- At INT8 the weights do move (6 of 7 methods change 55 to 96%), but no method reaches the forgetting target, and none does in full precision either with these settings.
- The way of quantizing does not matter. The cause is that one unlearning update is smaller than half a quantization step, so rounding undoes it.

## Table: methods that changed nothing (out of 7), per way of quantizing and size
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

## What this supports
See `GAPS.md`. In short: it supports the gap that no method exists to unlearn a quantized time-series model without a full-precision copy.

## What this does not show
- That no setting can ever work. A much larger learning rate can move the weights; we ran one fixed standard setting per method.
- That INT8 unlearning is correct. The weights move, the forgetting target is not reached.
- Results for other datasets, other forgotten clients, real hardware (the integers here are simulated), or methods outside the 7 tested.
- The methods are our simplified versions of the papers. Seeds: DLinear 3, MLP 2, PatchTST 1.
