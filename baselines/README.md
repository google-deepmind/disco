# Behavior & Capability Discovery Baselines

This directory implements the five heuristic LLM discovery baselines evaluated
in Sections 2, 3, and 6 of the paper, along with per-task adapters and e-BH
significance testing wrappers (Algorithm 1).

---

## 1. Heuristic Discovery Baselines

Each baseline is characterized along three axes:

* **Query selection**: drawn from a fixed category taxonomy vs. proposed by the
  discovery agent.
* **Adaptivity**: whether query selection adapts to responses observed so far.
* **Hypothesis synthesis**: emitted programmatically from empirical
  per-category rates vs. synthesized by the discovery agent.

| Baseline (Paper Name) | Code ID | Literature Origin | Query Selection | Adaptive? | Hypothesis Synthesis |
|---|---|---|---|---|---|
| **Self-Report (SR)** | `direct_asking` | Sheshadri et al. (2026) | Asks the target model directly in natural language to describe its own behaviors. | No | Discovery agent summarizes the self-report transcript into hypotheses. |
| **Uniform Probing (UP)** | `simple_probing` | Sheshadri et al. (2026) | Spreads budget evenly across a fixed taxonomy of task categories. | No | Programmatic — one hypothesis per category from observed rates. |
| **Repeated-Sample Probing (RSP)** | `looped_multisample` | Sheshadri et al. (2026) | Fixed taxonomy concentrated on fewer categories with $k \ge 2$ samples per query. | No | Discovery agent synthesizes holistically from a cumulative research log. |
| **Adaptive Red Teaming (ART)** | `red_team` | Perez et al. (2022) | Discovery agent proposes batches of edge-case queries conditioned on prior batch responses. | Yes (between batches) | Discovery agent synthesizes hypotheses from all observations at the end. |
| **Investigator Agent (IA)** | `looped_investigator` | Sheshadri et al. (2026) | Discovery agent maintains a cumulative research log across rounds to propose new queries. | Yes (every round) | Discovery agent synthesizes final hypotheses from the full research log. |

---

## 2. Running Discovery Baselines (`eval_discovery_baselines.py`)

Ensure `GEMINI_API_KEY` is set in your environment, then run:

```bash
# Example 1: Run all 5 baselines on Recipe Translation (--task_suite=recipes)
python -m disco.baselines.eval_discovery_baselines \
  --task_suite=recipes \
  --target_model=gemini-3.8-flash \
  --backbone_model=gemini-3.8-flash \
  --baseline_method=all \
  --budget=450 \
  --output_dir=./outputs/recipe_baselines

# Example 2: Run on Epistemic Hedging
python -m disco.baselines.eval_discovery_baselines \
  --task_suite=hedging \
  --target_model=gemini-3.8-flash \
  --backbone_model=gemini-3.8-flash \
  --baseline_method=all \
  --budget=300 \
  --output_dir=./outputs/hedging_baselines

# Example 3: Run on Visual Symmetry
python -m disco.baselines.eval_discovery_baselines \
  --task_suite=visual_symmetry \
  --target_model=gemini-3.1-flash-image \
  --backbone_model=gemini-3.8-flash \
  --baseline_method=all \
  --budget=400 \
  --output_dir=./outputs/visual_symmetry_baselines

# Example 4: The same, from a YAML config (flags on the command line win)
python -m disco.baselines.eval_discovery_baselines \
  --config=configs/recipes_baselines.yaml --budget=300
```

The baselines use the 25-category (400-item) category pools, dynamic $k$ for
repeated-sample probing, batched red-team generation and investigator rounds
that scale with the budget — the configuration used for the scaled budget
experiments in the paper.

Flags (everything worth setting per run):

| Flag | Default | Meaning |
|---|---|---|
| `--target_model` | `gemini-3.8-flash` | Model under audit. |
| `--backbone_model` | (empty) | Model for every auxiliary role not set explicitly: the investigator, the T2I judge, and the verifier's agent, parser, rating and summarizer models. |
| `--investigator_model` | `gemini-3.8-flash` | Discovery agent; falls back to `--backbone_model` when that is set. |
| `--task_suite`, `--task_name` | `recipes`, `all` | Task suite and task within it. |
| `--baseline_method` | `all` | `all`, `llm_methods`, `empirical_methods`, or a comma-separated list. |
| `--budget` | 20 | Stage 1 target-query budget per method. |
| `--max_predictions` | 10 | Hypotheses kept per method ($\bar{K}$ below). |
| `--output_dir`, `--experiment_name` | | Where results go. |
| `--seed_instruction` | (task's) | Overrides the task's seed brief. |
| `--replica` | 0 | Added to the sampling seed, for repeated runs. |
| `--alpha` | 0.05 | Stage 2 significance level. |
| `--num_queries` | 100 | Stage 2 verification queries per hypothesis. |
| `--total_verification_budget` | 0 | If set, Stage 2 budget split across all hypotheses (overrides `--num_queries`). |
| `--config` | | YAML file of flag values and YAML-only settings (see below). |

Every other setting is YAML-only: a `BaselineRunSettings` field in
`eval_discovery_baselines.py` (`run_precision_check`, `red_team_batch_size`,
`random_seed`, `procedural_autorater`,
`direct_asking_total_verification_budget`, `single_batch_null_prob`,
`concurrency`, `max_retries`, `base_retry_delay`) or a `VerifierConfig` field in
`verifier_config.py` (e.g. per-role `agent_model` / `parser_model` /
`rating_model` / `summarizer_model`, `max_concurrent_hypotheses`,
`scrape_bandwidth`). `run_precision_check` defaults to `true`, which runs
Stage 2 precision verification in-job after Stage 1 discovery completes
(spending `--total_verification_budget` or `--num_queries` per hypothesis
against the target model); set `run_precision_check: false` in the YAML for
discovery-only runs.

### Wrapping Baselines with e-BH (Algorithm 1)

To apply post-hoc e-BH FDR control (Algorithm 1 in Section 3.3 of the paper):

1. Run discovery with half the total query budget ($B_{\text{disc}} = B / 2$)
   and maximum output size $\bar{K}$ (`--max_predictions`).
2. Run significance testing on the discovered hypotheses $R_{\text{LLM}}$ with
   budget $B_{\text{sig}} / |R_{\text{LLM}}|$ per hypothesis to compute
   $p$-values $p^h$ and calibrated $e$-values $E^h = \kappa (p^h)^{\kappa - 1}$
   — either in-job via `--total_verification_budget=<B_sig> --alpha=<alpha>` in
   `eval_discovery_baselines.py`, or standalone via
   `python -m disco.standalone_precision_verifier` (passing
   `--hypotheses_path=<path/to/hypotheses.json>`, `--alpha=<alpha>`, and
   `--config=<verifier.yaml>`) with `total_verification_budget: <B_sig>` and
   `kappa: <kappa>` set in the YAML file (both are YAML-only `VerifierConfig`
   fields).
3. Apply post-hoc e-BH FDR control using `python -m disco.apply_fdr_control`
   with `--run_dir=<run_dir>`, `--alpha=<alpha>`, `--kappa=<kappa>`, and
   `--ebh_family_size=max_predictions` (which invokes `compute_ebh_unboosted`,
   `compute_ebh_boosted`, and `e_bh_rejection` in `stats/multiple_testing.py`
   with `stats/ebh_family.py`) to obtain the FDR-controlled rejection set $R$,
   written to `<run_dir>/ebh_fdr/`.

---

## 3. Code Structure

```
baselines/
├── README.md                    # This documentation
├── common.py                    # Target calls, checkpointing, investigator
├── methods/                     # 5 discovery methods + registry
├── adapters/                    # Per-task adapters + registry
├── data/                        # Category pools (.jsonl)
├── pool_loader.py               # JSONL loader for category pools in data/
└── eval_discovery_baselines.py  # CLI runner for unified discovery baselines
```
