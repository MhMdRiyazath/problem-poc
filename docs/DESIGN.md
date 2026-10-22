# Design (written before the quantize-and-unlearn runs)

## Question
If a time-series forecasting model has already been quantized, can the unlearning methods used in the time-series unlearning literature make it forget one client's data, with no full-precision copy and no retraining from scratch?

## Pipeline
1. Data: Electricity (hourly, 20 clients used). Clients 1 to 10 train the model, client 10 is the one to forget, clients 11 to 20 are never trained on (non-members for the attack). Look-back 168 h, forecast 24 h. Time split 70/10/20, per-client z-score from the training period.
2. Train, in full precision: an **original** model (all 10 clients) and a **retrained** model (client 10 left out, the ideal forgotten model). Models: DLinear, MLP, PatchTST. Seeds: DLinear 3, MLP 2, PatchTST 1 (training cost).
3. Quantize, post-training, to signed integers: weights of every Linear layer, round-to-nearest, bits 8, 6, 4, 3, 2. Three techniques: per-channel abs-max, per-tensor abs-max (one scale per layer, common on small hardware), per-channel with 99th-percentile clipping. Biases, LayerNorm and position tables stay in full precision. The original and the retrained model are quantized identically.
4. Unlearn the quantized original. No full-precision copy is used: only the Linear weights (integers) may change, after every step they are snapped back to the integer grid, scales stay fixed, biases and norms are frozen.
5. Check against the equally quantized retrained model.

## Methods (time-series unlearning literature only)
RandLabel, ZeroLabel, NoisyLabel, MaxLoss, TS-Unlearn dual objective (TSU), Gaussian Amnesiac, Blindspot. Simplified re-implementations from the paper descriptions.
Each runs **once with one fixed setting** (Adam, 400 steps, batch 64, lr 1e-3; MaxLoss 1e-4; Gaussian Amnesiac noise 0.1 then 200 repair steps at lr 3e-4). No search, no tuning, no choosing the best step: the model after the last step is judged.
The same setting is run in full precision as the reference for what the method does when quantization is not an obstacle.

## Measurements per run
- fraction of integer weights changed by unlearning
- forget score: 0 = still like the original, 1 = like the retrained model
- retain error ratio: retain-client test MSE divided by the start model's
- membership-inference AUC (0.5 = cannot tell client 10 from unseen clients)

## Rules fixed in advance
- **stuck**: fewer than 1% of the integer weights changed (the method did nothing).
- **same as FP**: not stuck, and forget score within 0.25 of the full-precision run of the same method and seed, and retain error at most 0.10 worse.
- **moved, different from FP**: otherwise.
- A cell (model, technique, bits, method) takes the state that holds in the majority of seeds.
- The strict rule (forget score 0.8 to 1.2 with retain error at most 10% worse) is also reported. It is only meaningful for methods that satisfy it in full precision.

## Limits
One dataset, one forgotten client, simulated integer weights (floats on the integer grid), simplified methods, one fixed setting per method (a different setting could behave differently), few seeds for MLP and PatchTST.
