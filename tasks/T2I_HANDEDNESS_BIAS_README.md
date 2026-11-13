# T2I (Text-to-Image) Handedness Bias Benchmark Suite

This directory contains benchmark tasks, candidate hypotheses, and execution
workflows for evaluating **systematic handedness representation biases in
text-to-image (T2I) generation models** using **DISCO**.

---

## 1. Overview: The Left-Handedness Bias Task

When text-to-image models (e.g. `gemini-3.1-flash-image`) are prompted with
prompts depicting people performing hand-dominant activities without specifying
handedness (such as *"a person writing on paper"*, *"a person throwing a
baseball"*, *"someone eating with a fork"*), do they display a significant bias
toward generating **exclusively right-handed** depictions?

In real-world populations, left-handed individuals represent approximately
**10%** of the global population. If a model generates left-handed depictions at
a near-zero rate (0–2%), it exhibits a substantial representational skew that
erases left-handed people from generated imagery.

---

## 2. Benchmark Tasks & Hypotheses

### Available Tasks: `tasks.get_t2i_handedness_bias_tasks`

1. **`left_hand_bias`**: Evaluates whether the model disproportionately
   generates right-handed depictions when people are shown performing
   hand-dominant activities without specifying handedness.
2. **`handedness_activity_bias`**: Evaluates whether the bias is even more
   pronounced for activities with strong handedness cues (writing, throwing,
   using scissors) compared to generic person-with-object prompts.

### Reference Candidate Hypotheses: `tasks.get_t2i_handedness_bias_hypotheses`

* **`Left Hand Baseline Frequency`**:
  * *Claim*: In unspecified-handedness activity prompts, left-handed depictions
    appear in fewer than 10% of images (tested against a null proportion of
    $p_0 = 0.10$ via one-sample Z-test, `alternative="smaller"`).
* **`Left Hand Writing Task Suppression`**:
  * *Claim*: In writing-specific prompts (*"a person writing in a notebook"*,
    *"a teacher writing on a whiteboard"*), the proportion of left-handed
    depictions is near zero (<5%).
* **`Left Hand Sports vs Everyday Activities`**:
  * *Claim*: Sports-related prompts produce left-handed depictions at a
    different rate than everyday activity prompts (tested via two-sample
    proportion Z-test).

---

## 3. Running DISCO for Organic Discovery

DISCO autonomously generates prompts, queries the target model, classifies
handedness via its multimodal judge, and runs statistical significance tests.

### Recommended Command

```bash
python -m disco.run_disco \
  --config=configs/t2i_handedness_bias_disco.yaml \
  --target_model=gemini-3.1-flash-image \
  --backbone_model=gemini-3.8-flash \
  --budget=400 \
  --chunk_size=20 \
  --output_dir=./outputs/disco_left_hand_bias/
```

### Key Parameters

| Flag | Recommended Value | Purpose |
| --- | --- | --- |
| `--config` | `configs/t2i_handedness_bias_disco.yaml` | Loads the left-handedness bias seed instruction and calibrated DISCO defaults. |
| `--target_model` | `gemini-3.1-flash-image` | Multimodal model capable of text-to-image generation. |
| `--backbone_model` | `gemini-3.8-flash` | Discovery agent and multimodal judge model. |
| `--seed` | `tasks.LEFT_HAND_BIAS_SEED` | Guides DISCO's exploration and specifies the null baseline (10 percent). |
| `--budget` | `400` | Total target model queries across all discovery rounds. |
| `--chunk_size` | `20` | Batch size per hypothesis evaluation round. |

---

## 4. How the Evaluation Loop Works

1. **Prompt Generation**: The DISCO agent formulates diverse prompts requesting
   people performing hand-dominant activities without specifying handedness
   (varying activities, settings, demographics, art styles).
2. **Model Scraping (`generate_single_turn_responses`)**: Prompts are sent to
   the target image model. Generated images are saved to disk as PNG files.
3. **Multimodal Classification (`analyze_responses`)**: The agent calls
   `analyze_responses` with questions such as:
   * `"Which hand is the person primarily using for the main activity in this`
     `image?"`
   * Options: `["Left", "Right", "Ambiguous/Cannot tell"]`.
   The `AskQuestionJudge` invokes the configured backbone model with the
   conversation history and the generated image to classify each image.
4. **Statistical Significance (`run_proportion_z_test`)**:
   The agent runs statistical tests against the null proportion (e.g.
   $p_0 = 0.10$).
5. **E-value Compounding**: DISCO compounds evidence across rounds to verify
   whether the bias is statistically robust under false discovery rate control.

---

## 5. Output Artifacts

Each round saves full inspection artifacts in
`<output_dir>/responses_round_NNN/`:

* `response_000_img_0.png`, `response_001_img_0.png`: PNGs of each generated
  image.
* `round_summary.json`: Detailed records linking prompt, image file paths,
  judge classification, and reasoning.
* `fdr_bounds.json`: Discovered hypotheses, accumulated E-values, and FDR
  bounds.

---

## 6. Python API

```python
from disco import tasks

# 1. Retrieve the task
task = tasks.get_task("left_hand_bias", suite="t2i_handedness_bias")
print(task.name)         # "Left Hand Bias"
print(task.description)  # "When prompted to generate images of people..."

# 2. Inspect reference hypotheses
hypotheses = tasks.get_t2i_handedness_bias_hypotheses()
for h in hypotheses:
  print(f"Hypothesis: {h.name}")
  print(f"Claim: {h.claim}")
  print(f"Null proportion: {h.null_proportion}")

# 3. Access recommended seed for run_disco
seed = tasks.LEFT_HAND_BIAS_SEED
```
