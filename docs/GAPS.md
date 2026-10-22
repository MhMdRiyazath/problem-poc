# Gaps and what this proof of concept shows

| Gap | Statement | What supports it here |
|---|---|---|
| **Core 1** | There is no framework to unlearn a quantized time-series model without retraining from scratch. | **Supported.** The literature on unlearning for time series (TS-Unlearn, task-aware unlearning) is full precision only. The literature on unlearning for quantized models is images and language models only. In our runs the time-series methods cannot change a quantized model below INT8. |
| **Core 2** | Existing unlearning methods are not shown to work on quantized time-series models. | **Supported.** 7 time-series methods, 3 models, 3 ways of quantizing, 5 sizes: below INT8 the weights do not change; at INT8 they change but the forgetting target is not reached. |
| **Supporting 1** | Fast, edge-feasible unlearning for time series is not addressed. | **Motivated, not measured.** The result shows a deployed quantized model cannot be updated with standard unlearning steps. Time and memory cost were not measured here. |
| **Supporting 2** | Checking that forgetting happened on the quantized model, without a retrained reference, is not addressed. | **Motivated, not measured.** Every check in this proof of concept compares against a retrained model, which a deployed device would not have. Reference-free checking was not tested. |

The first two gaps are tested by this work. The last two are the reasons the work matters and are left for the framework.
