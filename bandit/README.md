# `bandit/` — Contextual Bandit Synthetic Benchmark for DISCO

Monte Carlo simulation framework for Section 5 of the paper (*Experiments:
Synthetic*). Evaluates **DISCO** (pre-hoc and post-hoc bounds), **BUW-eBH**
(Bandit Uniform-Weighted e-BH), and **Type-I Error Control** in a
$d$-dimensional Gaussian-cluster contextual bandit environment.

---

## Directory Structure

| File | Role |
|---|---|
| `bandit_env.py` | Contextual and Bernoulli bandit environments where behaviors correspond to Gaussian clusters in $\mathbb{R}^d$ above a baseline noise rate $p_0$. |
| `bandit_disco.py` | Runner implementations: `run_disco_bandit`, `run_offline_ebh_bandit` (BUW-eBH), `run_type1_bandit`, and arm-selection policies (`DISCOContextualCandidateSelector`, `OfflineContextualCandidateSelector`, `choose_active_slot`). |
| `run_bandit_disco.py` | Main simulation CLI: assembles `BanditConfig` from flags, runs Monte Carlo trials, and writes summary and per-run JSON outputs. |
| `tune_bandit_disco.py` | Bayesian hyperparameter optimization CLI using open-source [`google-vizier`](https://github.com/google/vizier) (`vizier.service.clients`) to maximize cluster coverage under FDR control. |

---

## Running a Simulation (`run_bandit_disco.py`)

Run a side-by-side comparison (`--runner_type=sxs` for `offline_ebh` + `disco`,
or `--runner_type=all` for `offline_ebh` + `type1` + `disco`) or an individual
runner (`disco`, `offline_ebh`, `type1`):

```bash
python -m disco.bandit.run_bandit_disco \
  --runner_type=sxs \
  --env_type=contextual \
  --alpha=0.10 \
  --budget=900 \
  --chunk_size=30 \
  --num_mc_runs=200 \
  --output_path=./outputs/bandit_results.json
```

### JSON Output Schema

`run_bandit_disco.py` writes `<output_path>`:

```json
{
  "params": { "...BanditConfig / CLI flag fields..." },
  "summary": {
    "<runner_type>": {
      "cluster_coverage": {
        "mean": 0.0,
        "std": 0.0,
        "sem": 0.0,
        "ci95": 0.0,
        "ci_low": 0.0,
        "ci_high": 0.0
      },
      "fdr": {
        "mean": 0.0,
        "std": 0.0,
        "sem": 0.0,
        "ci95": 0.0,
        "ci_low": 0.0,
        "ci_high": 0.0
      }
    }
  },
  "bounds_report": {
    "nominal_calibrated_alpha": 0.0,
    "prehoc_unboosted": 0.0,
    "prehoc_boosted": 0.0,
    "prehoc_analytical_ceiling": 0.0,
    "posthoc_epsilon_mean": 0.0,
    "posthoc_delta_mean": 0.0,
    "posthoc_marginal_mean": 0.0
  },
  "alpha_sweep": []
}
```

By default (`--save_per_run_data=true`), per-trial arrays are also written to
`<name>_runs.json` (pass `--nosave_per_run_data` to disable).

---

## Bayesian Hyperparameter Tuning (`tune_bandit_disco.py`)

Requires `google-vizier` (`pip install -e ".[bandit]"`):

```bash
python -m disco.bandit.tune_bandit_disco \
  --runner_type=disco \
  --env_type=contextual \
  --alpha_calibration_mode=good_event \
  --alpha=0.05 \
  --budget=900 \
  --chunk_size=30 \
  --num_mc_runs=200 \
  --num_trials=200 \
  --output_dir=./outputs/vizier_bandit_study
```

### Key Flags

* `--runner_type`: Which bandit runner(s) to tune:
  * `disco`: DISCO (optimizes $\kappa$, $n_{\max}$, $e_{\max}$, $e_{\min}$, and
    active binning $\mathcal{P}$; choose between post-hoc / good-event /
    pre-hoc calibration via
    `--alpha_calibration_mode=post_hoc|good_event|pre_hoc`).
  * `offline_ebh`: BUW-eBH (optimizes $\kappa$, $n_{\max}$, and $e_{\max}$).
  * `type1`: Bandit Type-I error control ($e_{\max} = 1/\alpha$, optimizes
    $\kappa$ and $n_{\max}$).
  * `sxs` / `all`: Runs separate Vizier studies sequentially for `disco` +
    `offline_ebh` (or all three).
* `--algorithm`: Open-source Vizier search algorithm (`DEFAULT`, `GP_UCB_PE`,
  `GAUSSIAN_PROCESS_BANDIT`, `RANDOM_SEARCH`, `QUASI_RANDOM_SEARCH`).
* `--primary_metric`: Single objective maximized by Vizier (default
  `penalized_cluster_coverage`). For `disco`, excess post-hoc bad-event
  probability ($\delta_{\text{post}} > \text{delta\_threshold}$, default $0.05$)
  incurs a linear hinge penalty (`--delta_multiplier=5.0` per $0.01$ excess),
  and the study summary JSON records both `best_observed_trial` and
  `best_feasible_trial` ($\delta_{\text{post}} \le \text{delta\_threshold}$).
