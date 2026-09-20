# Alpha and scale study results

See `study.json` for the exact source directory used by each setting, `per_seed.csv` for paired final-round measurements, and `summary.csv` for means and sample standard deviations across seeds 42, 43, and 44.

Accuracy damage (percentage points) = 100 × (clean accuracy − poisoned accuracy).
Loss damage = poisoned loss − clean loss.
Positive values mean the attack made performance worse; negative values mean improvement. Accuracy columns use fractions (0–1). Damage is paired within each seed before averaging.

Experiment A changes alpha with scale fixed at 10. Experiment B changes scale with alpha fixed at 0.1. Their shared alpha=0.1, scale=10 setting refers to the same runs and is not independent evidence. Repeated clean controls across scale settings are also not additional independent seeds. Scale 1 is a control: ordinary updates should give approximately zero damage.

Three seeds provide preliminary variability estimates, not proof of statistical significance. Alpha also changes client sample counts and aggregation weights; interpret this as the effect of the whole partitioning scheme, not label imbalance alone.
