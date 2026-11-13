# DISCO Benchmark & Evaluation Tasks

This package defines the five benchmark task suites used in Section 6 of the
paper (two text-to-text suites and three multimodal / image-generation suites),
along with the unified task registry (`tasks/registry.py`) and YAML task
specification loader (`tasks/loader.py`).

---

## Available Task Suites

| Suite Key | Paper Task Name | Modality | Default Null $p_0$ | Description |
|---|---|---|---|---|
| `recipes` | **Word Count** | Text-to-Text | `0.10` (`recipe_translate_french_long`) | Discovers dish categories where English recipe generations are systematically shorter than French recipe generations for the same dish (`count_words` procedural autorater). |
| `hedging` | **Hedging** | Text-to-Text | `0.33` (`hedging_unhedged`) | Discovers contested or debated topic domains where the target model states claims without epistemic hedging markers (`count_hedges` procedural autorater). |
| `t2i_cat_bias` | **Cats** | Text-to-Image | `0.20` (`orange_cat_bias`) | Discovers prompt conditions where a text-to-image model disproportionately generates orange/ginger cats when coat color is unspecified. See [`T2I_CAT_BIAS_README.md`](T2I_CAT_BIAS_README.md). |
| `t2i_handedness_bias` | **Handedness** | Text-to-Image | `0.10` (`left_hand_bias`) | Evaluates whether a text-to-image model disproportionately depicts right-handed actions when prompted with hand-dominant activities without specifying handedness. See [`T2I_HANDEDNESS_BIAS_README.md`](T2I_HANDEDNESS_BIAS_README.md). |
| `visual_symmetry` | **Symmetry** | Multimodal (Image + Text $\to$ Image) | `0.90` (`visual_symmetry_completion`) | Evaluates whether a multimodal model can accurately reflect visual patterns across a central vertical axis across 22 pattern categories and 10 background colors. See [`visual_symmetry/README.md`](visual_symmetry/README.md). |

---

## Directory Structure

```
tasks/
├── README.md                        # This documentation
├── __init__.py                      # Top-level package exports
├── spec.py                          # Typed TaskSpec dataclass
├── loader.py                        # Loads suites from data/<suite>.yaml
├── data/                            # Task definitions (YAML)
├── data_classes.py                  # Task and Hypothesis dataclasses
├── registry.py                      # Centralized Task and Hypothesis registry
├── hedging_autorater.py             # Deterministic hedging autoraters
├── hedging_tasks.py                 # Hedging task constants & hypotheses
├── recipe_tasks.py                  # Recipe task constants & hypotheses
├── t2i_cat_bias_tasks.py            # T2I Cat Bias tasks & hypotheses
├── T2I_CAT_BIAS_README.md           # Guide for T2I Cat Bias tasks
├── t2i_handedness_bias_tasks.py     # T2I Handedness Bias tasks & hypotheses
├── T2I_HANDEDNESS_BIAS_README.md    # Guide for T2I Handedness Bias tasks
└── visual_symmetry/                 # Visual Symmetry task package
```

---

## Python API

```python
from disco import tasks
from disco.tasks import registry

# List all registered task suites
print(tasks.list_suites())
# ['hedging', 'recipes', 't2i_cat_bias', ...]

# Resolve the default TaskSpec for a suite
recipe_task = registry.resolve("recipes")
print(recipe_task.name)             # "Recipe Translate French Long"
print(recipe_task.null_proportion)  # 0.1

# Look up a specific task by key or alias
hedging_task = registry.get("hedging_unhedged")
print(hedging_task.seed_instruction)
```

---

## Adding Custom Task Suites

You can add a new task suite either via a YAML file in
`tasks/data/<suite>.yaml` (and adding `<suite>` to `BUILTIN_SUITES` in
`tasks/loader.py`) or programmatically at runtime:

```python
from disco import tasks

custom_tasks = [
    tasks.Task(
        name="Custom Task",
        description=(
            "Discover inputs where the target model exhibits behavior X."
        ),
        split="evaluation",
        suite="custom_suite",
        seed_instruction="Investigate when the model exhibits behavior X...",
        null_proportion=0.20,
    ),
]
tasks.register_suite("custom_suite", custom_tasks)
```
