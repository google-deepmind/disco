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

"""`--retry_failed`: rerun only the failed records of a verification result.

A verification run whose target had an outage leaves records with status
SCRAPE_FAILED or INSUFFICIENT_RESPONSES. Rather than rerun every hypothesis,
`standalone_precision_verifier --retry_failed` reruns just those, merges the
new records over the old ones, and recomputes the summary. This module holds
the pure parts (selection, merging, backup naming, history); the verifier does
the I/O.
"""

from collections.abc import Callable, Mapping, Sequence
import datetime
from typing import Any

from disco.verification import scoring

# Record statuses `--retry_failed` reruns.
DEFAULT_RETRY_STATUSES: tuple[str, ...] = (
    "SCRAPE_FAILED",
    scoring.INSUFFICIENT_RESPONSES_STATUS,
)


def select(
    records: Sequence[Mapping[str, Any]], statuses: Sequence[str]
) -> list[int]:
  """Returns the positions of the records whose status is in `statuses`."""
  wanted = set(statuses)
  return [
      i for i, r in enumerate(records) if scoring.record_status(r) in wanted
  ]


def merge(
    records: Sequence[Mapping[str, Any]],
    retried_positions: Sequence[int],
    new_records: Sequence[Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
  """Merges retried records over the originals.

  Each new record replaces one of the retried originals: the first with the
  same label, else the first with the same (whitespace-normalized) text. A new
  record that matches neither is dropped. Replaced records get `n_attempts`
  set to the original's (default 1) plus one.

  Args:
    records: The original records, in their saved order.
    retried_positions: Positions in `records` that were rerun.
    new_records: Records from the rerun.

  Returns:
    The merged records (original order) and one history entry per replaced
    record: label, status before and after.
  """
  merged = [dict(r) for r in records]
  open_positions = list(retried_positions)
  changes: list[dict[str, Any]] = []

  def _take(match: Callable[[dict[str, Any]], bool]) -> int | None:
    for pos in open_positions:
      if match(merged[pos]):
        open_positions.remove(pos)
        return pos
    return None

  for new in new_records:
    label = new.get("label")
    text = scoring.normalize_hypothesis(new.get("hypothesis_text", ""))
    pos = _take(lambda r, lbl=label: lbl is not None and r.get("label") == lbl)
    if pos is None:
      pos = _take(
          lambda r, txt=text: scoring.normalize_hypothesis(
              r.get("hypothesis_text", "")
          )
          == txt
      )
    if pos is None:
      continue
    old = merged[pos]
    merged[pos] = {**new, "n_attempts": int(old.get("n_attempts", 1)) + 1}
    changes.append({
        "label": old.get("label"),
        "status_before": scoring.record_status(old),
        "status_after": scoring.record_status(new),
    })
  return merged, changes


def backup_path(out_path: str, exists: Callable[[str], bool]) -> str:
  """Returns where to copy `out_path` before it is overwritten by a retry.

  `<stem>.pre_retry.json`, or `<stem>.pre_retry.<N>.json` with the smallest
  N >= 2 that does not exist yet.

  Args:
    out_path: The results file, e.g. `.../verification_results.json`.
    exists: File existence check.
  """
  stem, ext = (
      (out_path[: -len(".json")], ".json")
      if out_path.endswith(".json")
      else (out_path, "")
  )
  candidate = f"{stem}.pre_retry{ext}"
  n = 2
  while exists(candidate):
    candidate = f"{stem}.pre_retry.{n}{ext}"
    n += 1
  return candidate


def summary_counts(records: Sequence[Mapping[str, Any]]) -> dict[str, int]:
  """Counters the verifier writes next to the score, recomputed from records.

  Same definitions as `precision_estimation.run_precision_check`.

  Args:
    records: Verification records.

  Returns:
    Dictionary of total, evaluated, and failure record counts.
  """
  n_execution_errors = sum(1 for r in records if scoring.is_failed(r))
  return {
      "n_total": len(records),
      "n_evaluated": scoring.score(list(records)).n_scored,
      "n_statistical_failures": sum(
          1
          for r in records
          if not scoring.is_failed(r)
          and not scoring.record_field(r, "verified", False)
      ),
      "n_execution_errors": n_execution_errors,
      "n_verification_failures": n_execution_errors,
  }


def history_entry(
    statuses: Sequence[str],
    changes: Sequence[Mapping[str, Any]],
    *,
    now: datetime.datetime | None = None,
) -> dict[str, Any]:
  """One `retry_history` entry."""
  now = now or datetime.datetime.now()
  return {
      "timestamp": now.isoformat(timespec="seconds"),
      "retry_statuses": list(statuses),
      "n_retried": len(changes),
      "n_recovered": sum(
          1 for c in changes if c["status_after"] == scoring.OK_STATUS
      ),
      "records": [dict(c) for c in changes],
  }
