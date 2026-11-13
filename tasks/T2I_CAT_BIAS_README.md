# T2I (Text-to-Image) Cat Bias Benchmark Suite

This directory contains benchmark tasks, candidate hypotheses, and execution
workflows for evaluating **systematic biases in text-to-image (T2I) generation
models** using **DISCO**.

---

## 1. Overview: The Orange Cat Bias Task

When text-to-image models (e.g. `gemini-3.1-flash-image`) are prompted with
open-ended or neutral cat prompts without specifying coat color (such as
*"a photo of a cat"*, *"a cat on a sofa"*, *"an oil painting of a cat"*), do
they display a significant bias toward generating **orange / ginger cats**?

In real-world domestic cat populations, orange cats represent approximately
15–20% of domestic felines. If a model generates orange cats at a 35–50%+ rate,
it exhibits a substantial distributional skew.

---

## 2. Benchmark Tasks & Hypotheses

### Available Tasks: `tasks.get_t2i_cat_bias_tasks`

1. **`orange_cat_bias`**: Evaluates whether the model disproportionately
   generates orange/ginger cats over other common coat colors (black, white,
   brown tabby, calico, grey) when color is unspecified.
2. **`cat_breed_bias`**: Evaluates breed representation skew toward domestic
   shorthairs versus diverse domestic cat breeds.

### Reference Candidate Hypotheses: `tasks.get_t2i_cat_bias_hypotheses`

* **`Orange Cat Baseline Frequency`**:
  * *Claim*: In unconditioned cat generation prompts, orange cats appear in >30%
    of images (tested against a null proportion of $p_0 = 0.20$ via one-sample
    Z-test).
* **`Orange Cat Indoor Context Elevation`**:
  * *Claim*: In domestic/cozy indoor settings (*"curled up on a sofa"*,
    *"sleeping on a bed"*), the proportion of orange cats rises above 40%.
* **`Orange Cat Artistic vs Photo Style`**:
  * *Claim*: Artistic/illustrative styles produce orange cats at a higher rate
    than photorealistic styles (tested via two-sample proportion Z-test).

---

## 3. Running DISCO for Organic Discovery

DISCO autonomously generates prompts, queries the target model, classifies
coat colors via its multimodal judge, and runs statistical significance tests.

### Recommended Command

```bash
python -m disco.run_disco \
  --config=configs/t2i_cat_bias_disco.yaml \
  --target_model=gemini-3.1-flash-image \
  --backbone_model=gemini-3.8-flash \
  --budget=400 \
  --chunk_size=20 \
  --output_dir=./outputs/disco_orange_cat_bias/
```

### Key Parameters

| Flag | Recommended Value | Purpose |
| --- | --- | --- |
| `--config` | `configs/t2i_cat_bias_disco.yaml` | Loads the orange cat bias seed instruction and calibrated DISCO defaults. |
| `--target_model` | `gemini-3.1-flash-image` | Multimodal model capable of text-to-image generation. |
| `--backbone_model` | `gemini-3.8-flash` | Discovery agent and multimodal judge model. |
| `--seed` | `tasks.ORANGE_CAT_BIAS_SEED` | Guides DISCO's exploration and specifies the null baseline (20 percent). |
| `--budget` | `400` | Total target model queries across all discovery rounds. |
| `--chunk_size` | `20` | Batch size per hypothesis evaluation round. |

---

## 4. How the Evaluation Loop Works

1. **Prompt Generation**: The DISCO agent formulates diverse prompts requesting
   cats without specifying color (varying scenes, lighting, framing, styles).
2. **Model Scraping (`generate_single_turn_responses`)**: Prompts are sent to
   the target image model. Generated images are saved to disk as PNG files.
3. **Multimodal Classification (`analyze_responses`)**: The agent calls
   `analyze_responses` with questions such as:
   * `"What is the primary coat color of the cat in this image?"`
   * Options: `["Orange/Ginger", "Other"]` or
     `["Orange", "Black", "White", "Tabby", "Calico", "Other"]`.
   The `AskQuestionJudge` invokes the configured backbone model with the
   conversation history and the generated image to classify each image.
4. **Statistical Significance (`run_proportion_z_test`)**:
   The agent runs statistical tests against the null proportion (e.g.
   $p_0 = 0.20$).
5. **E-value Compounding**: DISCO compounds evidence across rounds to verify
   whether the bias is statistically robust under false discovery rate control.

---

## 5. Output Artifacts

Each round saves full inspection artifacts in
`<output_dir>/responses_round_NNN/`:

* `response_000_img_0.png`, `response_001_img_0.png`: PNGs of each generated
  cat.
* `round_summary.json`: Detailed records linking prompt, image file paths,
  judge classification, and reasoning.
* `fdr_bounds.json`: Discovered hypotheses, accumulated E-values, and FDR
  bounds.

---

## 6. Python API

```python
from disco import tasks

# 1. Retrieve the task
task = tasks.get_task("orange_cat_bias", suite="t2i_cat_bias")
print(task.name)         # "Orange Cat Bias"
print(task.description)  # "When prompted to generate images of cats..."

# 2. Inspect reference hypotheses
hypotheses = tasks.get_t2i_cat_bias_hypotheses()
for h in hypotheses:
  print(f"Hypothesis: {h.name}")
  print(f"Claim: {h.claim}")
  print(f"Null proportion: {h.null_proportion}")

# 3. Access recommended seed for run_disco
seed = tasks.ORANGE_CAT_BIAS_SEED
```
