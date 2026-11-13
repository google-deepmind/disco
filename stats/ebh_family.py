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

"""The e-BH family size K: which hypotheses count toward the multiplicity.

e-BH controls FDR only when K is fixed before the data are seen. The agent
proposes a data-dependent number of hypotheses, so the pre-registered cap
(`max_predictions` in the run's flags.json) is the valid choice: every slot the
agent did not fill, and every hypothesis whose test failed, behaves exactly
like a hypothesis carrying e = 0, which is a valid e-variable that can never be
rejected.

Modes (`--ebh_family_size`):
  max_predictions: K = cap. Failed and unfilled slots both enter at e = 0.
  max_predictions_minus_failed: K = cap - (failed records). Failed records
    leave the family; unfilled slots still enter at e = 0.
  observed: K = the number of executed records. Adaptive, so the guarantee is
    void; kept for comparison with older figures.

Failed records never enter e-BH with their stored p-value: a calibrated e at
p = 1 is kappa > 0, not 0, so feeding them would not be "e = 0".
"""

from collections.abc import Mapping, Sequence
import functools
import json
import logging
import os
from typing import Any

MAX_PREDICTIONS = "max_predictions"
MAX_PREDICTIONS_MINUS_FAILED = "max_predictions_minus_failed"
OBSERVED = "observed"

MODES: tuple[str, ...] = (
    MAX_PREDICTIONS,
    MAX_PREDICTIONS_MINUS_FAILED,
    OBSERVED,
)
DEFAULT_MODE = MAX_PREDICTIONS

HELP = (
    "Which family size K the two e-BH procedures use. e-BH controls FDR only"
    " when K is fixed in advance, but the agent proposes a data-dependent"
    " number of hypotheses. 'max_predictions' (default) uses the run's"
    " pre-registered cap from flags.json, which is equivalent to giving every"
    " unfilled slot and every failed record e = 0."
    " 'max_predictions_minus_failed' removes failed records from the family"
    " (K = cap - failed). 'observed' uses the number of executed records,"
    " which makes K adaptive and voids the guarantee. Runs whose flags.json"
    " declares no cap fall back to 'observed' with a warning. BH and BY are"
    " unaffected."
)

# How far above an input file to look for the run's flags.json. The deepest
# supported layout is <run>/<Method>/<verification_subdir>/<file>, so three
# levels reaches the run directory. Deliberately bounded: an open-ended search
# up the tree would issue unpredictable numbers of filesystem stats per target.
MAX_PREDICTIONS_SEARCH_LEVELS = 3


def validate_mode(mode: str) -> str:
  """Returns `mode` if it is one of `MODES`.

  Args:
    mode: A candidate `--ebh_family_size` value.

  Raises:
    ValueError: If `mode` is not a known mode.
  """
  if mode not in MODES:
    raise ValueError(f"ebh_family_size must be one of {MODES}, got {mode!r}")
  return mode


@functools.lru_cache(maxsize=4096)
def read_max_predictions(dir_path: str) -> int | None:
  """Reads max_predictions from a single directory's flags.json, if present."""
  flags_path = os.path.join(dir_path, "flags.json")
  try:
    if not os.path.exists(flags_path):
      return None
    with open(flags_path, "r", encoding="utf-8") as f:
      value = json.load(f).get("max_predictions")
  except (
      OSError,
      json.JSONDecodeError,
      ValueError,
      TypeError,
      AttributeError,
  ) as e:
    logging.warning("Could not read %s: %s", flags_path, e)
    return None
  try:
    parsed = int(value)
  except (TypeError, ValueError):
    return None
  return parsed if parsed > 0 else None


def find_max_predictions(input_path: str) -> int | None:
  """Finds the pre-registered hypothesis cap for a target's run.

  Both `<run>/flags.json` and `<run>/<Method>/flags.json` carry
  `max_predictions`, so the nearest one above the input file is used. Results
  are cached per directory.

  Args:
    input_path: Path to the target's verification or evaluation file.

  Returns:
    The cap, or None when no flags.json above the input declares one.
  """
  current = os.path.dirname(input_path)
  for _ in range(MAX_PREDICTIONS_SEARCH_LEVELS):
    if not current or current in ("/", "."):
      break
    found = read_max_predictions(current)
    if found is not None:
      return found
    current = os.path.dirname(current)
  return None


def resolve(
    input_path: str,
    mode: str = DEFAULT_MODE,
    max_predictions_override: int = 0,
    n_failed: int = 0,
) -> tuple[int | None, str]:
  """Decides the family size K that e-BH should use for one target.

  Args:
    input_path: Path to the target's input file; its flags.json supplies the
      cap.
    mode: One of `MODES`.
    max_predictions_override: Explicit cap; 0 means read flags.json.
    n_failed: Records whose test failed. Only `max_predictions_minus_failed`
      reads it.

  Returns:
    Tuple of (family size, or None to use the observed count; source label).
    The label is 'observed', 'flag' or 'flags_json' (with a '_minus_failed'
    suffix in that mode), or 'observed_fallback' when a cap was wanted but
    could not be found.

  Raises:
    ValueError: If `mode` is not one of `MODES`.
  """
  validate_mode(mode)
  if mode == OBSERVED:
    return None, "observed"
  if max_predictions_override > 0:
    cap, source = max_predictions_override, "flag"
  else:
    discovered = find_max_predictions(input_path)
    if discovered is None:
      return None, "observed_fallback"
    cap, source = discovered, "flags_json"
  if mode == MAX_PREDICTIONS_MINUS_FAILED:
    return max(cap - max(n_failed, 0), 0), source + "_minus_failed"
  return cap, source


def failed_slots_warning(
    where: str,
    failed_records: Sequence[Mapping[str, Any]],
    mode: str,
    k: int | None,
) -> str:
  """One log line naming every failed record and how e-BH treated it.

  Args:
    where: The run or target the records belong to.
    failed_records: The records whose test failed (non-empty).
    mode: The `--ebh_family_size` mode, or "" when the procedure is not e-BH (BH
      and BY simply leave failed records out).
    k: The resolved family size, or None when the observed count is used.

  Returns:
    The warning text.
  """
  if not mode:
    treatment, setting = "excluded from the family", ""
  elif mode == MAX_PREDICTIONS and k is not None:
    treatment = f"counted in K={k} at e=0"
    setting = f" (--ebh_family_size={mode})"
  else:
    treatment = "excluded from the e-BH family"
    setting = f" (--ebh_family_size={mode})"
  slots = "; ".join(
      f"label={r.get('label')!r} status={r.get('status') or 'unset'}"
      f" reason={r.get('failure_reason') or r.get('error') or '-'!r}"
      for r in failed_records
  )
  return (
      f"[e-BH] {where}: {len(failed_records)} failed record(s) {treatment}"
      f"{setting}: {slots}"
  )
