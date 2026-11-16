# DISCO: Reliable Behavior Discovery via False Discovery Rate Control

This codebase implements the **DISCO** (*Discovery through Iterative Search with
COarsened $e$-processes*) algorithm and
baselines from the paper *Reliable Behavior Discovery via False Discovery
Rate Control* (Madras et al., 2026).

---

## Installation & Setup

### 1. Create a Virtual Environment & Install Dependencies

From the root of this repository:

```bash
python3 -m venv .venv
source .venv/bin/activate

pip install -e .

# Optional: install open-source Google Vizier and dev/test dependencies
pip install -e ".[bandit,dev]"
```

For rendering CJK, Greek, and Latin glyphs in the **Visual Symmetry** task
suite, install a CJK-capable system font (such as `fonts-noto-cjk`) or point
`DISCO_FONT_PATH` to a TrueType/OpenType font file:

```bash
# Ubuntu / Debian
sudo apt-get update && sudo apt-get install -y fonts-noto-cjk

# macOS (Homebrew)
brew install --cask font-noto-sans-cjk

# Or specify a custom font path directly:
export DISCO_FONT_PATH="/path/to/NotoSansCJK-Regular.ttc"
```

### 2. Run Unit Tests & Smoke Check (No API Key Required)

You can verify your installation immediately using the hermetic test suite and
synthetic bandit runner:

```bash
pytest
python -m disco.bandit.run_bandit_disco \
  --env_type=bernoulli \
  --runner_type=all \
  --budget=90 \
  --chunk_size=15 \
  --num_mc_runs=3
```

### 3. Configure Gemini API Key

All generative model experiments use the official
[`google-genai`](https://pypi.org/project/google-genai/) SDK. Export
`GEMINI_API_KEY` (or `GOOGLE_API_KEY`) before running discovery, baseline, or
verification scripts:

```bash
export GEMINI_API_KEY="your-gemini-api-key"
# Alternatively:
export GOOGLE_API_KEY="your-gemini-api-key"
```

---

## Directory Structure

* **Repository Root**:
  * `run_disco.py`: Main CLI entry point for running DISCO, BUW-eBH
    (`method: ebh_baseline` in the YAML config), and Type-I error control
    (`method: type1_baseline`) on generative model tasks.
  * `eval_disco.py`: End-to-end discovery + out-of-sample precision/FDR
    verification wrapper around `run_disco.py`.
  * `standalone_precision_verifier.py`: Out-of-sample single-batch $p$-value
    verification runner for evaluating candidate hypotheses from any discovery
    method.
  * `apply_fdr_control.py`: Post-hoc multi-method FDR control (BH, BY,
    unboosted/boosted e-BH) and Storey $\hat{\pi}_0$ / thresholded FDR
    estimation on verification outputs.
  * `compute_disco_bound_values.py`: Monte Carlo utility for computing pre-hoc
    and post-hoc DISCO FDR bounds ($\epsilon$, $\delta$, $\mathcal{V}^*(N)$, and
    $\alpha^\star$).
  * `configs/`: Single-run YAML configuration files for the paper's benchmark
    experiments (`recipes_disco.yaml`, `hedging_disco.yaml`,
    `t2i_cat_bias_disco.yaml`, `t2i_handedness_bias_disco.yaml`,
    `visual_symmetry_broad_disco.yaml`,
    `visual_symmetry_contrastive_disco.yaml`, and `recipes_baselines.yaml`).
  * `example_hypotheses.json`: Example candidate hypotheses file for standalone
    verification.
  * `runners.py`: Core `DiscoRunner` orchestrating the iterative bandit-style
    discovery loop.
  * `backbone_client.py` & `models/`: Gemini client (`google-genai`) and model
    registry (`gemini-3.8-flash` for text tasks, `gemini-3.1-flash-image` for
    image tasks).
  * `core/`: Configuration (`config.py`, `flag_specs.py`, `run_config.py`),
    checkpoints, round execution, task execution, and finalizers (DISCO
    pre-hoc/post-hoc, BUW-eBH, Type-I).
  * `stats/`: $p$-to-$e$ calibrators, $e$-value boosting, e-BH / BH / BY
    multiple testing procedures, statistical tests (binomial, Fisher's exact,
    McNemar's, Student's $t$), Storey $\hat{\pi}_0$ and thresholded FDR
    estimators, BCa bootstrap confidence intervals, and `stats/coarsened/` null
    random-walk tables and certificates.
  * `agent/`, `prompting/`, `parsing/`: Discovery agent loop, taxonomy-based
    query generation, structured output schemas, and fault-tolerant target
    scraping.
  * `verification/`, `eval/`: Hypothesis I/O, task resolution, FDR/precision
    scoring, and evaluation result schemas.
  * `testing/`: Hermetic unit, golden, and smoke test suites.
* **[`tasks/`](tasks/README.md)**: Benchmark task suites for generative model
  behavior discovery:
  1. **Word Count (`recipes`)**: Cross-lingual recipe response length
     comparisons (English vs. French).
  2. **Epistemic Hedging (`hedging`)**: Unhedged controversial claims and
     epistemic calibration.
  3. **T2I Cat Coat Color Bias (`t2i_cat_bias`)**: Distributional coat-color
     bias in text-to-image generation.
  4. **T2I Handedness Bias (`t2i_handedness_bias`)**: Right-handedness
     representation skew in text-to-image generation.
  5. **Visual Symmetry (`visual_symmetry`)**: Bilateral spatial reflection
     completion across 22 pattern categories and 10 background colors.
* **[`baselines/`](baselines/README.md)**: The 5 heuristic LLM discovery
  baselines and their e-BH wrappers:
  * Self-Report (`direct_asking`)
  * Uniform Probing (`simple_probing`)
  * Repeated-Sample Probing (`looped_multisample`)
  * Adaptive Red Teaming (`red_team`)
  * Investigator Agent (`looped_investigator`)
* **[`bandit/`](bandit/README.md)**: Contextual bandit synthetic environment
  (`bandit_env.py`), bandit DISCO / BUW-eBH / Type-I runners
  (`bandit_disco.py`, `run_bandit_disco.py`), and Bayesian hyperparameter
  optimization via open-source Vizier (`tune_bandit_disco.py`).

---

## Quick Start

### 1. Synthetic Contextual Bandit Experiments (Section 5)

Run a Monte Carlo comparison across DISCO (`--alpha_calibration_mode=post_hoc`
by default, or `pre_hoc`), BUW-eBH (`offline_ebh`), and Type-I error control
(`type1`). Use `--runner_type=sxs` for BUW-eBH vs. DISCO or `--runner_type=all`
to include Type-I:

```bash
python -m disco.bandit.run_bandit_disco \
  --runner_type=all \
  --env_type=contextual \
  --alpha=0.10 \
  --budget=900 \
  --chunk_size=30 \
  --num_mc_runs=100 \
  --output_path=./outputs/bandit_all.json
```

Run Bayesian hyperparameter optimization using `google-vizier`:

```bash
python -m disco.bandit.tune_bandit_disco \
  --runner_type=disco \
  --alpha=0.10 \
  --budget=900 \
  --chunk_size=30 \
  --num_trials=200 \
  --output_dir=./outputs/bandit_vizier_tuning
```

See [`bandit/README.md`](bandit/README.md) for full details.

---

### 2. Generative Model Behavior Discovery with DISCO (Section 6)

You can launch any of the paper's benchmark experiments from a single-run YAML
configuration file in `configs/` using `--config` (explicit CLI flags override
values in the YAML file). A YAML key may name either a command-line flag or any
field of the run's config dataclass (`DiscoConfig` for `eval_disco` /
`run_disco`, `VerifierConfig` for the verifiers), so settings without a flag of
their own can still be set from YAML:

```bash
# Text tasks (gemini-3.8-flash)
python -m disco.eval_disco --config=configs/recipes_disco.yaml
python -m disco.eval_disco --config=configs/hedging_disco.yaml

# Image tasks (gemini-3.1-flash-image)
python -m disco.eval_disco --config=configs/t2i_cat_bias_disco.yaml
python -m disco.eval_disco --config=configs/t2i_handedness_bias_disco.yaml
python -m disco.eval_disco --config=configs/visual_symmetry_broad_disco.yaml
python -m disco.eval_disco \
  --config=configs/visual_symmetry_contrastive_disco.yaml
```

Or override the core flags on the command line (`eval_disco` and `run_disco`
share the same flag names: `--target_model`, `--seed`, `--budget`, `--alpha`,
`--backbone_model`, `--output_dir`, `--name`, `--e_max`, `--chunk_size`,
`--alpha_calibration_mode`, `--n_max`, `--num_active_bins`):

```bash
python -m disco.eval_disco \
  --config=configs/recipes_disco.yaml \
  --target_model=gemini-3.8-flash \
  --backbone_model=gemini-3.8-flash \
  --budget=900 \
  --chunk_size=30 \
  --alpha=0.10 \
  --alpha_calibration_mode=post_hoc \
  --output_dir=./outputs/disco_runs \
  --name=word_count_disco
```

* Every other `DiscoConfig` setting (e.g. `task_suite`, `kappa`, `agent_model`,
  `query_generation_mode`, `paired`, `run_precision_check`,
  `single_batch_num_queries`) has no flag and is set in the YAML file.
* Set `--alpha_calibration_mode=post_hoc` (or `good_event`) for DISCO (post-hoc
  bound) or `--alpha_calibration_mode=pre_hoc` for DISCO (pre-hoc bound). For
  BUW-eBH set `method: ebh_baseline` in the YAML, and for Type-I Error Control
  set `method: type1_baseline` (the default `method: disco` runs DISCO).
* For image generation tasks (`t2i_cat_bias`, `t2i_handedness_bias`,
  `visual_symmetry`), pass `--target_model=gemini-3.1-flash-image`.

---

### 3. Heuristic LLM Baselines (Sections 2 & 3)

Run all 5 heuristic LLM discovery baselines (`direct_asking`, `simple_probing`,
`looped_multisample`, `red_team`, `looped_investigator`) on any task suite via
a YAML config or CLI flags:

```bash
# Using a YAML config (with optional CLI overrides):
python -m disco.baselines.eval_discovery_baselines \
  --config=configs/recipes_baselines.yaml

# Or using CLI flags directly:
python -m disco.baselines.eval_discovery_baselines \
  --task_suite=recipes \
  --target_model=gemini-3.8-flash \
  --backbone_model=gemini-3.8-flash \
  --baseline_method=all \
  --budget=450 \
  --output_dir=./outputs/baseline_runs
```

See [`baselines/README.md`](baselines/README.md) for details on wrapping
baseline hypotheses with e-BH (Algorithm 1).

---

### 4. Standalone Out-of-Sample Verification & Post-Hoc FDR Control

Verify any JSON file of candidate hypotheses (such as
[`example_hypotheses.json`](example_hypotheses.json) or a `fdr_bounds.json`
produced by `run_disco.py`) using fresh out-of-sample target queries ($N = 100$
by default) and compute both thresholded ($p \le 0.05$) and Storey
$\hat{\pi}_0$ FDR estimates:

```bash
python -m disco.standalone_precision_verifier \
  --hypotheses_path=example_hypotheses.json \
  --task_suite=recipes \
  --target_model=gemini-3.8-flash \
  --backbone_model=gemini-3.8-flash \
  --num_queries=100 \
  --alpha=0.05 \
  --output_dir=./outputs/verification_out
```

The verifier's flags are `--hypotheses_path`, `--task_suite`,
`--seed_instruction`, `--output_dir`, `--experiment_name`, `--target_model`,
`--backbone_model` (the agent, parser, rating and summarizer models),
`--alpha`, `--num_queries` and `--retry_failed` (rerun only the failed records
of an existing result). Every other `VerifierConfig` setting (e.g. per-role
`agent_model` / `parser_model` / `rating_model` / `summarizer_model`, `paired`,
`p_value_mode`, `total_verification_budget`, `random_seed`,
`procedural_autorater`, `max_concurrent_hypotheses`, `scrape_bandwidth`) has no
flag and is set in a `--config` YAML file.

Apply post-hoc BH, BY, and unboosted/boosted e-BH multiple-testing corrections
(along with Storey $\hat{\pi}_0$ and thresholded FDR estimates) to any
completed verification or evaluation directory:

```bash
python -m disco.apply_fdr_control \
  --run_dir=./outputs/verification_out \
  --alpha=0.05
```

`apply_fdr_control` takes `--run_dir` (or `--input_path` naming a specific
`verification_results.json` / `evaluation_results.json`), `--source` (`auto`,
`evaluation` or `verification`), `--alpha`, `--kappa`, `--ebh_family_size` and
`--max_predictions` (the pre-registered cap when the run has no `flags.json`),
and writes `verification_results_fdr.json`, `fdr_summary.json` and
`hypotheses_ranked.csv` to an `ebh_fdr/` directory beside the input file.

For the **Visual Symmetry** task suite, use the dedicated continuous symmetry
verifier (`tasks/visual_symmetry/verify_visual_symmetry.py`, whose flags are
`--experiment_dir`, `--hypotheses_path`, `--output_dir`, `--target_model`,
`--backbone_model`, `--num_probes`, `--alpha`, `--null_accuracy` and
`--iou_threshold`); see
[`tasks/visual_symmetry/README.md`](tasks/visual_symmetry/README.md).

---

## Citing This Work

If you use this code or benchmark suites in your research, please cite:

```
@article{madras2026disco,
  title   = {Reliable Behavior Discovery via False Discovery Rate Control},
  author  = {Madras, David and Wang, Zi and Andreetto, Marco and
             Burke, Ken and Pantofaru, Caroline and Prabhu, Utsav},
  journal = {arXiv preprint},
  year    = {2026},
}
```

---

## License and Disclaimer

Copyright 2026 Google LLC

All software is licensed under the Apache License, Version 2.0 (Apache 2.0);
you may not use this file except in compliance with the Apache 2.0 license. You
may obtain a copy of the Apache 2.0 license at:
https://www.apache.org/licenses/LICENSE-2.0

All other materials are licensed under the Creative Commons Attribution 4.0
International License (CC-BY). You may obtain a copy of the CC-BY license at:
https://creativecommons.org/licenses/by/4.0/legalcode

Unless required by applicable law or agreed to in writing, all software and
materials distributed here under the Apache 2.0 or CC-BY licenses are
distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND,
either express or implied. See the Licenses for the specific language governing
permissions and limitations under those licenses.

This is not an official Google product.
