# Visual Symmetry Completion Task Suite

This directory contains a visual symmetry completion task suite for evaluating
**spatial reasoning capabilities** of multimodal image generation models. This
suite is designed for **organic discovery** — DISCO explores the model's
capabilities and proposes hypotheses about which visual patterns and background
conditions the model can or cannot successfully mirror.

---

## Task Description

The target model (e.g. `gemini-3.1-flash-image`)
receives a continuous visual image with a pattern on the left half and a blank
right half. The model must complete the right half so that the final image is
perfectly symmetric along the central vertical reflection axis while preserving
the left half.

**Success criterion** (default `both`): a trial succeeds if the returned image
is symmetric about its own centre (tolerant-F1 self-mirror score $\ge 0.90$)
**and** keeps the left half it was given (left fidelity $\ge 0.80$). See
[Evaluation Metrics](#evaluation-metrics) for the other criteria.

---

## Hypothesis Format & Reference Hypotheses

DISCO discovers hypotheses of the form:

> *"The model can perform vertical symmetry completion on **[condition]**
> succeeding at least 90% of the time."*

Where **[condition]** specifies a shape type (such as `english_letter`,
`chinese_char`, `square`, `circle`, `stripes`, `snowflake`, etc.) or background
color.

### Candidate Reference Hypotheses in `hypotheses.py`

The entries in [`hypotheses.py`](hypotheses.py) are **predefined candidate
hypotheses** (`Hypothesis` in `tasks/data_classes.py`) across the task's search
space. They assert conditions like *"the model completes symmetry on English
letters $\ge 90\%$ of the time"*, providing standardized reference hypotheses
for hypothesis verification, organic capability discovery, and comparative
baseline evaluation.

---

## Search Space

### Shape types

| Shape Type | Description |
|-----------|-------------|
| `english_letter` | Symmetric English letters (A, H, I, M, O, T, U, V, X, Y) |
| `greek_letter` | Symmetric Greek letters (Α, Δ, Θ, Λ, Ξ, Π, Τ, Υ, Φ, Ψ, Ω) |
| `chinese_char` | Symmetric Chinese characters (中, 十, 王, 田, 八, 木, 土, 丰, 工, 口, 日, 三, 干, 丫, 甲, 由, 申) |
| `square` | Solid colored square of varying sizes |
| `circle` | Solid colored circle of varying radii |
| `triangle` | Centered isosceles triangle |
| `plus` / `cross` | Centered perpendicular cross / plus sign |
| `heart` | Centered symmetric heart curve |
| `stripes` | Horizontal stripe patterns |
| `dots` | Sparse dot patterns |
| `random_sparse` | Sparse ink patterns (~10% density) |
| `random_dense` | Dense ink patterns (~50% density) |
| `complex_chinese` | Multi-stroke symmetric Chinese characters (囍, 業, 晶, 串, 冊, 皿, 凹, 凸, 亞, 巫, 幽, 韭, 非, 鼎, 束) |
| `multi_glyph` | Vertically stacked 3-glyph columns |
| `maze` | Rectilinear mazes |
| `snowflake` | 6-fold radial snowflakes with sub-branches |
| `filigree` | Scattered filigree arcs and ticks |
| `nested_outlines` | Concentric nested outlines |
| `starburst` | Radial starbursts |
| `scatter_rings` | Scattered hollow rings |
| `wave_lines` | Stacked sinusoidal wave lines |
| `fine_noise` | High-frequency fine stochastic noise |
| `dynamic` | Procedural symmetric polygon generator (`"dynamic"` is a built-in shape type; pass `allow_dynamic=True` to accept arbitrary unknown shape names) |

---

## Running with DISCO (Organic Hypothesis Discovery)

DISCO uses specialized visual symmetry tools to explore the model's
capabilities:

| Tool | What It Does |
|------|-------------|
| `generate_and_send_symmetry_probes` | Generates N continuous test images $\rightarrow$ sends to target model $\rightarrow$ stores outputs |
| `score_symmetry_completions` | Scores outputs with tolerant-F1 ink matching under the hypothesis's `success_criterion` (default: self-mirror $\ge 0.90$ and left fidelity $\ge 0.80$) |
| `generate_symmetry_test_image` | Generates a single test image for inspection |

### Example command

```bash
python -m disco.run_disco \
  --config=configs/visual_symmetry_broad_disco.yaml \
  --target_model=gemini-3.1-flash-image \
  --backbone_model=gemini-3.8-flash \
  --output_dir=./outputs/disco_visual_symmetry/ \
  --budget=400 \
  --chunk_size=50
```

### Key flags

| Flag | Value | Description |
| --- | --- | --- |
| `--config` | `configs/visual_symmetry_broad_disco.yaml` | Loads `task_suite: visual_symmetry`, the seed instruction, and calibrated DISCO defaults |
| `--target_model` | `gemini-3.1-flash-image` | Target image model to test |
| `--backbone_model` | `gemini-3.8-flash` | Discovery agent model |
| `--seed` | *(optional override)* | Tells the agent what to explore; the phrase `"90 percent as the null"` is parsed by `stats_utils.extract_null_proportion_from_seed` to set the null proportion for statistical tests |
| `--budget` | `400` | Total queries across all rounds |
| `--chunk_size` | `50` | Queries per hypothesis round |

---

## Running Comparative Baselines

In addition to DISCO, you can run all 5 literature-grounded discovery
baselines on the visual symmetry suite via the unified
`eval_discovery_baselines` CLI:

- Self-Report (`direct_asking`)
- Uniform Probing (`simple_probing`)
- Repeated-Sample Probing (`looped_multisample`)
- Adaptive Red Teaming (`red_team`)
- Investigator Agent (`looped_investigator`)

```bash
python -m disco.baselines.eval_discovery_baselines \
  --task_suite=visual_symmetry \
  --target_model=gemini-3.1-flash-image \
  --investigator_model=gemini-3.8-flash \
  --baseline_method=all \
  --output_dir=./outputs/visual_symmetry_baselines/
```

---

## Out-of-Sample Hypothesis Verification (`verify_visual_symmetry`)

To evaluate the empirical replication rate and calculate the Empirical False
Discovery Rate for hypotheses discovered across methods, run the verification
CLI against an experiment output directory (from `run_disco` or
`eval_discovery_baselines`):

```bash
python -m disco.tasks.visual_symmetry.verify_visual_symmetry \
  --experiment_dir=./outputs/disco_visual_symmetry/ \
  --target_model=gemini-3.1-flash-image \
  --backbone_model=gemini-3.8-flash \
  --alpha=0.05 \
  --null_accuracy=0.90 \
  --iou_threshold=0.90
```

The verifier's flags are `--experiment_dir`, `--hypotheses_path`,
`--output_dir`, `--target_model`, `--backbone_model` (the LLM that plans each
hypothesis's probes), `--num_probes` (default 100), `--alpha`,
`--null_accuracy` and `--iou_threshold`. The success criterion (the self-mirror
conjunction with its left-fidelity guard), the $p$-to-$e$ calibrator $\kappa$
and the request pacing are fixed in `verify_visual_symmetry.py` so that
verification tests the same proposition discovery did. Hypotheses already
verified in the output directory's registry are skipped; ones whose previous
attempt produced no usable measurement are retried.

You can also directly verify hypotheses from an exported JSON file (such as
`fdr_bounds.json`, `stage1_discovery_results.json`, or a custom JSON list):

```bash
python -m disco.tasks.visual_symmetry.verify_visual_symmetry \
  --hypotheses_path=./outputs/disco_visual_symmetry/fdr_bounds.json \
  --output_dir=./outputs/disco_visual_symmetry/verification/ \
  --target_model=gemini-3.1-flash-image \
  --backbone_model=gemini-3.8-flash
```

---

## Python API

### Generating test images

```python
from disco.tasks.visual_symmetry import image_gen as visual_symmetry_image_gen

# Generate a symmetric English letter canvas with the right half erased
gt_image, prompt_image, _ = visual_symmetry_image_gen.generate_symmetry_image(
    shape_type="english_letter",
    seed=42,
    fg_color="blue",
    canvas_size=320,
)

# gt_image:     (320, 320, 3) uint8 -- full reflectionally symmetric image
# prompt_image: (320, 320, 3) uint8 -- right half erased with subtle midline

# List all supported shape types
print(visual_symmetry_image_gen.list_shape_types())
```

### Scoring model output

```python
from disco.tasks.visual_symmetry import eval as visual_symmetry_eval

result = visual_symmetry_eval.score_symmetry_completion(
    output_image=model_output,       # (320, 320, 3) uint8
    ground_truth_cells=gt_image,     # (320, 320, 3) uint8
    iou_threshold=0.90,
)
is_ok = result["is_success"]
```

---

## Code Structure

```
tasks/
├── data_classes.py                # Task and Hypothesis dataclasses
├── data/visual_symmetry.yaml      # Visual Symmetry Task spec
└── visual_symmetry/
    ├── image_gen.py               # Matplotlib/PIL image generator
    ├── eval.py                    # Tolerant-F1 symmetry scoring
    ├── hypotheses.py              # Visual Symmetry Task + hypotheses
    ├── prompts.py                 # Hypothesis / success-criterion parsing
    ├── verify_visual_symmetry.py  # Out-of-sample verification CLI
    └── tasks_test.py              # Unit tests for visual symmetry suite

disco/
├── standalone_tools.py            # Tool declarations (+ 3 symmetry tools)
├── standalone_agent.py            # Tool implementations (+ symmetry dispatch)
├── core/task_executor.py          # SymmetryTaskExecutor
└── baselines/adapters/visual_symmetry.py  # Discovery-baselines adapter
```

---

## Evaluation Metrics

The scorer (`score_direct_symmetry_completion` in [`eval.py`](eval.py)) always
computes **three independent criteria** and reports all three. Which one decides
`is_success` is chosen by `success_mode`, and the agent selects it per
hypothesis via the `success_criterion` argument of `score_symmetry_completions`.

| `success_criterion` (agent) | `SUCCESS_MODE_*` (internal) | Success when |
|---|---|---|
| `preservation` | `PRESERVATION` | `left_fidelity >= 0.80` — the model kept the left half it was given |
| `symmetry` | `SYMMETRY` | `iou_self_mirror >= 0.90` — the returned image is symmetric about its own centre |
| `reconstruction` | `GROUND_TRUTH` | `iou_vs_ground_truth >= 0.90` — the right half matches the intended reflection |
| `both` *(default)* | `SELF_MIRROR` | `symmetry` **and** `preservation` |

---

## Running Tests

```bash
pytest tasks/visual_symmetry/tasks_test.py
```
