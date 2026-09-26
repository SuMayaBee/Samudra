<!--
SPDX-FileCopyrightText: 2026 Samudra Authors

SPDX-License-Identifier: CC-BY-4.0
-->

# ConvNeXt stochastic-depth ablation

This experiment asks whether dropping ConvNeXt residual branches improves the
Samudra U-Net, and whether a constant rate or a depth-dependent linear schedule
works better.

The search compares a no-stochastic-depth control with constant and linear
schedules at maximum rates of 0.1 and 0.2. All candidates use the same short
2011--2013 slice of the public two-degree OM4 demo data, model configuration,
random seed, optimizer, and training budgets. The existing U-Net shortcut
dropout remains disabled, so the experiment isolates residual-branch
stochastic depth. This is a screening experiment sized to finish within one
Kaggle T4 session, not a final model-quality result.

Successive halving trains every stochastic-depth candidate for one epoch,
promotes the strongest candidates to two epochs, and retains at least two
through the three-epoch rung. The fixed control runs for the full three epochs.

Run locally with:

```bash
python -m samudra.search experiments/convnext_stochastic_depth/search.yaml
```

The search writes resolved configs, checkpoints, metrics, provenance, and its
generated comparison report beneath `.LOCAL/searches/`.

The primary decision metric is validation loss. Before drawing a modeling
conclusion, also compare training loss, stability, runtime, and the final
validation gap between the control and surviving constant/linear candidates.
