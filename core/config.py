# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Configuration dataclass and prompt-module resolution for DISCO."""

from collections.abc import Mapping
import dataclasses
from typing import Any, Literal

from disco import behavior_discovery_prompts
from disco.core import flag_specs
from disco.core import prompts_protocol
from disco.models import registry as model_registry
from disco.parsing import parsers as parsers_lib
from disco.tasks import registry as task_registry
from disco.verification import scoring

Method = Literal["disco", "ebh_baseline", "type1_baseline"]
METHODS: tuple[str, ...] = ("disco", "ebh_baseline", "type1_baseline")


def get_prompts_module() -> prompts_protocol.FDRPromptsModule:
  """Returns the behavior-discovery prompts module."""
  return behavior_discovery_prompts


def _registry_null_proportion(task_suite: str) -> float | None:
  """Returns the registry task's null proportion for `task_suite`, if any."""
  if not task_suite:
    return None
  try:
    task = task_registry.resolve(task_suite)
  except (KeyError, ValueError):
    return None
  null_prop = getattr(task, "null_proportion", None)
  if null_prop is None and hasattr(task, "metadata"):
    null_prop = task.metadata.get("null_proportion")
  return None if null_prop is None else float(null_prop)


@dataclasses.dataclass
class DiscoConfig:
  """Full configuration for a single Disco red-teaming run.

  Intended to be constructed once in main() from FLAGS and the `--config`
  YAML, then passed through to run_loop — keeping FLAGS access isolated at
  the CLI boundary. Only the fields in `flag_specs.FLAG_SPECS` have a flag;
  every other field is set from the YAML file.
  """

  # Target and task
  target_model: str
  seed_instruction: str

  # Statistical parameters
  budget: int
  alpha: float
  # Kappa parameter for e-value conversion. The default maximizes the p-value
  # yielding E=20.
  kappa: float = 0.1741

  # Which procedure the run uses:
  # 'disco' - DISCO with online e-BH.
  # 'ebh_baseline' - unrestricted exploration + offline e-BH with fixed K_bar
  #                  (K_bar = budget // chunk_size).
  # 'type1_baseline' - Type-I error rate baseline: freeze hypotheses at
  #                    1/alpha, no e-BH at finalization.
  method: Method = "disco"

  # Auditor models. An empty `agent_model` resolves to `backbone_model`, or to
  # the default Gemini model; `parser_model` falls back to `agent_model`,
  # `rating_model` to the backbone/default and `summarizer_model` to
  # `rating_model` (see `__post_init__`).
  agent_model: str = ""
  agent_thinking_level: str | None = None
  agent_thinking_budget: int | None = None
  backbone_model: str = ""

  # Logging
  output_dir: str = ""
  experiment_name: str = "test"
  log_to_console: bool = True

  # Parser model
  parser_model: str = ""
  rating_model: str = model_registry.DEFAULT_GEMINI_MODEL
  summarizer_model: str = ""

  # Validation
  validate_stats_test: bool = False

  # Deactivation and boosting parameters. Defaults are the values the paper
  # runs used.
  e_min: float = 0.99
  e_max: float = 50.0
  enforce_exit_low: bool = True
  use_boosting: bool = True
  chunk_size: int = 25

  # Post-hoc alpha sweep for FDR bounds (computed at finalization).
  alpha_sweep_min: float = 0.01
  alpha_sweep_max: float = 0.99
  alpha_sweep_steps: int = 40

  # DISCO runner parameters (used by DiscoRunner only).
  # One of 'pre_hoc', 'post_hoc', 'good_event', 'nominal'.
  alpha_calibration_mode: str = "good_event"
  n_max: int = 3  # Max probes per hypothesis before timeout deactivation.
  # 0 = unbinned/exact, 1 = single active bin [e_min, e_max), >1 = multi-bin.
  num_active_bins: int = 1
  n_sim: int = 100_000  # Monte Carlo samples for null table precomputation.
  # Whether the DISCO probe agent sees failure examples from prior
  # sub-hypotheses within the same hypothesis (coarsening-safe).
  show_intra_hypothesis_failures: bool = True
  # If True, failure summaries are shown everywhere in the DISCO runner:
  # selector, hypothesis generation, within-hypothesis and cross-hypothesis.
  # This is coarsening-compliant only with `num_active_bins=0` (unbinned
  # e-values, where the agent already sees the exact e-values); with binned
  # e-values it leaks information the coarsening is meant to hide.
  show_all_failures: bool = False

  # Task suite this run belongs to (e.g. "visual_symmetry"). Lets verification
  # select the probe modality explicitly rather than inferring it from
  # `seed_instruction`.
  task_suite: str = ""

  # Reference / null success rate for this task, from the registry task's
  # `metadata["null_proportion"]`. 0.0 means unset.
  task_null_proportion: float = 0.0

  # Procedural autorater (empty = use LLM judge, or name of registered
  # autorater).
  procedural_autorater: str = ""
  allow_continuous_means_test: bool = True

  # Query generation mode:
  # 'standard' - Original flat query generation (default).
  # 'taxonomy' - Taxonomy-driven reasoning-driven generation
  #            (arXiv:2603.29791). Generates a hierarchical taxonomy of factors
  #            for the hypothesis, samples node combinations, converts them to
  #            diverse "meta prompts", optionally complexifies a fraction, and
  #            applies critic-based filtering.
  query_generation_mode: Literal["standard", "taxonomy"] = "standard"

  # Taxonomy-driven query generation parameters (arXiv:2603.29791).
  taxonomy_depth: int = 3  # Depth of the taxonomy tree to generate.
  taxonomy_use_critic: bool = True  # Whether to apply critic-based filtering.
  taxonomy_best_of_n: int = (
      3  # Number of taxonomy proposals to sample (Best-of-N).
  )
  # Whether to enforce balanced (A, B) query pairing across all modes.
  paired: bool = False
  use_direction_guard: bool = (
      True  # Whether to apply direction guard on p-values.
  )
  p_value_mode: Literal["randomized", "discrete", "discrete_matched"] = (
      "randomized"
  )
  # Seed of the p-value randomization RNGs (discovery and verification).
  # None seeds them from OS entropy.
  random_seed: int | None = None
  # Baseline options (only read when `method` is a baseline).
  baseline_enforce_exit_low: bool = False
  # Baseline mode: freeze a hypothesis as H once its running e-value reaches
  # K_bar / alpha (its e-BH rejection is then certain, so further probes are
  # wasted budget). Off by default to keep older baseline runs reproducible.
  baseline_exit_high: bool = False

  # Hypothesis compliance checker
  enable_hypothesis_checker: bool = True
  hypothesis_checker_max_retries: int = 2
  hypothesis_checker_rules: list[str] = dataclasses.field(default_factory=list)
  isolated_query_gen_temperature: float = 1.0
  # Out-of-sample precision verification of the discoveries (`eval_disco`).
  run_precision_check: bool = True
  single_batch_num_queries: int = 100
  single_batch_alpha: float = 0.05
  # Null baseline of the single-batch precision check. None (or a value <= 0)
  # extracts it per hypothesis.
  single_batch_null_prob: float | None = None
  # Concurrency.
  max_concurrent_hypotheses: int = 1
  scrape_bandwidth: int = 4
  # `--min_response_fraction` (`verification/min_response_fraction_flag`).
  min_response_fraction: float = scoring.DEFAULT_MIN_RESPONSE_FRACTION
  # Consecutive target-side or short skips before the discovery loop raises
  # `scrape_tolerance.TargetOutageError` (see `scrape_tolerance.RoundGuard`).
  # 0 disables the cap.
  max_consecutive_skips: int = 20

  # Derived from `method` in `__post_init__`.
  # Baseline mode: unrestricted exploration + offline e-BH with fixed K_bar.
  baseline_mode: bool = dataclasses.field(init=False, default=False)
  # Baseline Type-I error rate mode: freeze at 1/alpha, no e-BH at finalization.
  baseline_type1_mode: bool = dataclasses.field(init=False, default=False)

  def __post_init__(self):
    if self.method not in METHODS:
      raise ValueError(
          f"Unknown method {self.method!r}; expected one of {METHODS}."
      )
    self.baseline_mode = self.method == "ebh_baseline"
    self.baseline_type1_mode = self.method == "type1_baseline"
    if (
        self.show_all_failures
        and self.num_active_bins > 0
        and not self.baseline_mode
        and not self.baseline_type1_mode
    ):
      raise ValueError(
          "show_all_failures=True is not permitted when num_active_bins > 0"
          " in DISCO mode; set num_active_bins=0 or"
          " show_all_failures=False."
      )
    if (
        self.single_batch_null_prob is not None
        and self.single_batch_null_prob <= 0.0
    ):
      self.single_batch_null_prob = None

    default_model = model_registry.DEFAULT_GEMINI_MODEL
    if self.backbone_model:
      if not self.agent_model or self.agent_model == default_model:
        self.agent_model = self.backbone_model
      if not self.rating_model or self.rating_model == default_model:
        self.rating_model = self.backbone_model
      if not self.parser_model or self.parser_model == default_model:
        self.parser_model = self.agent_model
      if not self.summarizer_model or self.summarizer_model == default_model:
        self.summarizer_model = self.rating_model
    else:
      if not self.agent_model:
        self.agent_model = default_model
      if not self.rating_model:
        self.rating_model = default_model
      if not self.parser_model:
        self.parser_model = self.agent_model
      if not self.summarizer_model:
        self.summarizer_model = self.rating_model
    # Normalize every model name (aliases, `models/` prefixes) so a value set
    # from a YAML config is treated the same as one resolved from a flag.
    for model_field in (
        "target_model",
        "agent_model",
        "parser_model",
        "rating_model",
        "summarizer_model",
    ):
      setattr(
          self,
          model_field,
          model_registry.resolve_model_url(getattr(self, model_field)),
      )

  @property
  def is_paired(self) -> bool:
    """Whether this run forces paired A/B collection on every hypothesis.

    Note this is the *run-level* floor: with it false a hypothesis can still
    go paired on its own, via its pre-registered test plan.
    """
    return bool(self.paired)

  resolved_paired = is_paired

  def get_prompts_module(self) -> prompts_protocol.FDRPromptsModule:
    """Returns the prompts module (DISCO only does behavior discovery)."""
    return get_prompts_module()

  @classmethod
  def from_flags(
      cls,
      disco_flags: Any,
      *,
      log_to_console: bool = True,
      validate_stats_test: bool = False,
      experiment_name: str | None = None,
      output_dir: str | None = None,
      min_response_fraction: float = scoring.DEFAULT_MIN_RESPONSE_FRACTION,
      overrides: Mapping[str, Any] | None = None,
  ) -> "DiscoConfig":
    """Constructs a DiscoConfig from registered flag holders.

    Every flag in `flag_specs.FLAG_SPECS` is read from the holder of the same
    name on `disco_flags`, then the keyword arguments below are applied, then
    `overrides` (the YAML-only fields). Every other field keeps its dataclass
    default. `task_null_proportion`, if unset, is looked up in the task
    registry for `task_suite`.

    Args:
      disco_flags: Holders from `flags_lib.define_disco_flags`, or any object
        with `.value` attributes of the same names.
      log_to_console: Whether to also log to the console.
      validate_stats_test: Whether to validate the stats tool call.
      experiment_name: Overrides the `name` flag.
      output_dir: Overrides the `output_dir` flag.
      min_response_fraction: The `--min_response_fraction` value; that flag is
        defined outside `disco_flags`, so callers pass it in.
      overrides: Field name -> value for fields set from a YAML config rather
        than a flag (see `run_config.apply_yaml_config`). Applied after every
        flag has been read.

    Raises:
      ValueError: If `overrides` names something that is not a field.
    """
    kwargs: dict[str, Any] = {}
    for flag_name, spec in flag_specs.FLAG_SPECS.items():
      kwargs[spec.field_name(flag_name)] = getattr(disco_flags, flag_name).value
    kwargs["log_to_console"] = log_to_console
    kwargs["validate_stats_test"] = validate_stats_test
    kwargs["min_response_fraction"] = min_response_fraction
    if experiment_name is not None:
      kwargs["experiment_name"] = experiment_name
    if output_dir is not None:
      kwargs["output_dir"] = output_dir
    if overrides:
      field_names = {f.name for f in dataclasses.fields(cls) if f.init}
      unknown = sorted(name for name in overrides if name not in field_names)
      if unknown:
        raise ValueError(f"Unknown DiscoConfig field(s): {unknown}")
      kwargs.update(overrides)
    kwargs["hypothesis_checker_rules"] = list(
        kwargs.get("hypothesis_checker_rules") or []
    )
    if not kwargs.get("task_null_proportion"):
      null_prop = _registry_null_proportion(kwargs.get("task_suite", ""))
      if null_prop is not None:
        kwargs["task_null_proportion"] = null_prop
    return cls(**kwargs)

  def build_parsers(self) -> parsers_lib.Parsers:
    """Builds and returns a Parsers bundle using the configured parser model.

    Returns:
      A Parsers instance containing initialized parser instances.
    """
    return parsers_lib.Parsers.from_model(
        self.parser_model, validate_stats_test=self.validate_stats_test
    )
