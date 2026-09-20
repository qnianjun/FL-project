# Research design: alpha and attack scale

Goal: measure how data partitioning and model-update amplification affect federated learning, using matched clean and poisoned runs.

| Experiment | Variable | Values | Fixed setting |
| --- | --- | --- | --- |
| A | Alpha | 0.01, 0.1, 1, 10, 100 | Poison scale = 10 |
| B | Poison scale | 1, 2, 5, 10, 20 | Alpha = 0.1 |

Every setting uses seeds 42, 43, and 44. Each seed runs both clean and poisoned conditions with matched initial weights, data partition, and training shuffle seeds. Keep MNIST, five clients, client 0 as the attacker, ten rounds, one local epoch, batch size 32, and SGD learning rate 0.01 fixed.

Scale 1 is a control: it submits an ordinary update and should produce approximately zero damage. An increased scale does not guarantee increased damage.

## Run using the shell

From the project directory, preview the plan (no training):

```bash
bash run_study.sh --reuse-existing
```

Run both experiments:

```bash
bash run_study.sh --reuse-existing --run
```

This uses `.venv/bin/python`, validates the existing alpha 0.01 and 0.1 / scale 10 experiments, and runs the remaining seven unique settings: **42 new training runs**. The two experiments share alpha 0.1 / scale 10, so those runs are referenced in both analyses, not rerun. A completely fresh study has nine unique settings and 54 runs, versus 60 if the shared setting were needlessly repeated.

To run A or B separately:

```bash
bash run_study.sh --experiment A --reuse-existing --output results/studies/experiment_A --run
bash run_study.sh --experiment B --reuse-existing --output results/studies/experiment_B --run
```

To start a completely fresh study, omit `--reuse-existing` and choose a new `--output` directory. Existing completed settings are validated and reused when the same command is repeated. Existing incomplete or incompatible output folders cause a stop rather than an overwrite; preserve them and use a new output directory if needed. Configuration, source, dependency, and dataset checks prevent silently mixing incompatible saved runs.

The existing `run.sh`, `run_all.sh`, and `run_all.py` remain available. `run_study.sh` is a new wrapper for this research design, using the existing matched runner. It does not edit `config.json` or launch network servers.

## Outputs and attack damage

Default study folder: `results/studies/alpha_scale/`.

- `study.json`: which saved experiment provides each setting.
- `per_seed.csv`: final accuracy, loss, and paired damage for each seed.
- `summary.csv`: means and sample standard deviations across three seeds.
- `README.md`: metric definitions and interpretation limits.
- Each newly run setting has its own configurations, raw results, update norms, checksums, `REPORT.md`, `accuracy.png`, and `loss.png`. Reused experiments stay at their original paths, recorded in `study.json`.

**Accuracy damage (percentage points) = 100 × (clean accuracy − poisoned accuracy).**

**Loss damage = poisoned loss − clean loss.**

Positive damage means worse performance; negative damage means improvement. Calculate damage within each seed, then summarize the three paired differences. Do not treat rounds, repeated clean controls, or the shared A/B setting as extra independent seeds.

Alpha changes both label distribution and client sample counts. The attacker's FedAvg weight can therefore change across alpha values and seeds. Record this limitation when interpreting the alpha experiment. Three seeds give preliminary variability estimates; these two sweeps do not test every alpha–scale combination or establish statistical significance by themselves.
