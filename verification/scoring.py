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

"""Precision / empirical-FDR scoring of verification records.

Every writer and reader of a precision or FDR figure scores records with one
rule, `score`:

  * A record *failed* when its test produced no usable measurement: its status
    is not "OK" (see `FAILED_STATUSES`), or it predates the `status` field and
    has `sample_size <= 0`.
  * `failed_records="drop"` (the default) leaves failed records out of both the
    numerator and the denominator. `"count_as_false"` keeps them in the
    denominator as false discoveries, which reproduces earlier writers that did
    not filter on status.
  * precision = n_verified / n_scored, or None when nothing was scored; the
    empirical FDR is 1 - precision, or 0.0 when nothing was scored.

Which records are in scope (the discovery set) is the caller's choice and is
applied before `score`.

Records may be dicts (serialized JSON) or objects with attributes
(precision_estimation records).
"""

from collections.abc import Callable, Iterable, Mapping, Sequence
import dataclasses
import enum
from typing import Any

OK_STATUS = "OK"

# Statuses written by the verifiers for a record whose test produced no usable
# measurement. Any other non-OK status is treated the same way.
#   SCRAPE_FAILED: no (or too few) target responses were collected.
#   INSUFFICIENT_RESPONSES: fewer responses than --min_response_fraction.
#   PLAN_INCONSISTENT: the executed test could not be interpreted; the stored
#     p-value is a sentinel.
FAILED_STATUSES: frozenset[str] = frozenset(
    {"SCRAPE_FAILED", "INSUFFICIENT_RESPONSES", "PLAN_INCONSISTENT"}
)

# The status assigned to a record that predates the `status` field and was
# never measured.
LEGACY_UNMEASURED_STATUS = "SCRAPE_FAILED"

INSUFFICIENT_RESPONSES_STATUS = "INSUFFICIENT_RESPONSES"

# Default of `--min_response_fraction`: a verification test (or a discovery
# round) that received fewer than this fraction of the responses it requested
# is not scored.
DEFAULT_MIN_RESPONSE_FRACTION = 0.9


def is_short(n_received: int, n_requested: int, min_fraction: float) -> bool:
  """Whether `n_received` falls short of `min_fraction` of `n_requested`.

  Args:
    n_received: Usable responses (or complete pairs) received.
    n_requested: Responses (or pairs) requested. Nothing is short when 0.
    min_fraction: `--min_response_fraction`; 0 disables the check.

  Returns:
    True if `n_received` is below `min_fraction * n_requested`.
  """
  return n_requested > 0 and n_received < min_fraction * n_requested


def short_batch_reason(
    *,
    is_paired: bool,
    q_count: int,
    n_responses: int,
    n_pairs: int,
    min_fraction: float,
) -> str:
  """Why a batch is too short to score under `--min_response_fraction`.

  Args:
    is_paired: Whether the batch was collected as (A, B) pairs. Pairs are then
      the unit: `q_count // 2` requested against `n_pairs` complete.
    q_count: Queries sent (or planned).
    n_responses: Usable responses received.
    n_pairs: Complete pairs received (paired batches only).
    min_fraction: `--min_response_fraction`.

  Returns:
    "" if the batch may be scored, else e.g. "INSUFFICIENT_RESPONSES: 80/100
    responses received (< 0.9 required)".
  """
  if is_paired:
    n_req, n_rcv, unit = q_count // 2, n_pairs, "complete pairs"
  else:
    n_req, n_rcv, unit = q_count, n_responses, "responses"
  if not is_short(n_rcv, n_req, min_fraction):
    return ""
  return (
      f"{INSUFFICIENT_RESPONSES_STATUS}: {n_rcv}/{n_req} {unit}"
      f" received (< {min_fraction:g} required)"
  )


def short_arms_reason(
    arms: Sequence[tuple[str, int, int]], min_fraction: float
) -> str:
  """Why a two-arm batch of independent samples is too short to score.

  Each arm must reach `min_fraction` of the probes allocated to it. Both arms
  are load-bearing, as both halves of a pair are for `short_batch_reason`:
  pooling them would let a comparison be scored with one arm nearly empty.

  Args:
    arms: (label, usable responses received, probes requested) per arm.
    min_fraction: `--min_response_fraction`; 0 disables the check.

  Returns:
    "" if the batch may be scored, else e.g. "INSUFFICIENT_RESPONSES: arm 'B'
    6/10 responses received (< 0.9 required per arm)".
  """
  short = [
      f"arm '{label}' {n_rcv}/{n_req}"
      for label, n_rcv, n_req in arms
      if is_short(n_rcv, n_req, min_fraction)
  ]
  if not short:
    return ""
  return (
      f"{INSUFFICIENT_RESPONSES_STATUS}: {', '.join(short)} responses"
      f" received (< {min_fraction:g} required per arm)"
  )


class FailedRecords(enum.Enum):
  """How failed records enter a precision/FDR figure."""

  DROP = "drop"
  COUNT_AS_FALSE = "count_as_false"


FAILED_RECORDS_CHOICES: tuple[str, ...] = tuple(m.value for m in FailedRecords)
FAILED_RECORDS_HELP = (
    "How verification records whose test produced no usable measurement"
    " (status SCRAPE_FAILED, INSUFFICIENT_RESPONSES or PLAN_INCONSISTENT) enter"
    " precision and empirical FDR. 'drop' leaves them out of both numerator"
    " and denominator; 'count_as_false' counts them as false discoveries."
)


def parse_failed_records(value: "FailedRecords | str") -> FailedRecords:
  """Returns `value` as a FailedRecords member.

  Args:
    value: A member or its string value.

  Raises:
    ValueError: If `value` names no member.
  """
  if isinstance(value, FailedRecords):
    return value
  try:
    return FailedRecords(value)
  except ValueError:
    raise ValueError(
        f"failed_records must be one of {FAILED_RECORDS_CHOICES}, got {value!r}"
    ) from None


@dataclasses.dataclass(frozen=True)
class Score:
  """Result of scoring a set of verification records.

  Attributes:
    n_verified: Numerator of precision.
    n_scored: Denominator of precision.
    n_failed: Failed records among the inputs (scored or not).
    precision: n_verified / n_scored, or None when n_scored is 0.
    empirical_fdr: 1 - precision, or 0.0 when n_scored is 0.
    failed_by_status: Sorted (status, count) pairs over the failed records.
    failed_records: The mode the score was computed under.
    scored: The records that make up the denominator.
  """

  n_verified: int
  n_scored: int
  n_failed: int
  precision: float | None
  empirical_fdr: float
  failed_by_status: tuple[tuple[str, int], ...] = ()
  failed_records: FailedRecords = FailedRecords.DROP
  scored: tuple[Any, ...] = dataclasses.field(default=(), compare=False)

  def summary_fields(self) -> dict[str, Any]:
    """The JSON fields every precision/verification payload carries."""
    return {
        "precision": self.precision,
        "empirical_fdr": self.empirical_fdr,
        "n_verified": self.n_verified,
        "n_scored": self.n_scored,
        "n_failed": self.n_failed,
        "n_failed_by_status": dict(self.failed_by_status),
        "failed_records": self.failed_records.value,
    }

  def describe_failed(self) -> str:
    """One-line description of the failed records, for logs."""
    by_status = ", ".join(f"{s}={n}" for s, n in self.failed_by_status)
    action = (
        "excluded from precision/FDR"
        if self.failed_records is FailedRecords.DROP
        else "counted as false discoveries"
    )
    return f"{self.n_failed} failed record(s) ({by_status}) {action}"


def record_field(record: Any, name: str, default: Any = None) -> Any:
  """Reads `name` from a dict record or an attribute-style record."""
  if isinstance(record, Mapping):
    return record.get(name, default)
  return getattr(record, name, default)


def normalize_hypothesis(text: Any) -> str:
  """Collapses whitespace so texts match across writers."""
  return " ".join(str(text).split())


def record_status(record: Any) -> str:
  """Returns the record's status, inferring it for legacy records.

  A record without a `status` field is OK unless `sample_size <= 0`, in which
  case it was never measured and counts as `LEGACY_UNMEASURED_STATUS`.

  Args:
    record: A verification record.
  """
  status = record_field(record, "status")
  if status is not None:
    return str(status)
  sample_size = record_field(record, "sample_size")
  if sample_size is not None and sample_size <= 0:
    return LEGACY_UNMEASURED_STATUS
  return OK_STATUS


def is_failed(record: Any) -> bool:
  """True when the record's test produced no usable measurement."""
  return record_status(record) != OK_STATUS


def format_precision(precision: float | None) -> str:
  """Formats a precision for logs; 'n/a' when nothing was scored."""
  return "n/a" if precision is None else f"{precision:.4f}"


def score(
    records: Iterable[Any],
    *,
    failed_records: FailedRecords | str = FailedRecords.DROP,
) -> Score:
  """Scores in-scope verification records under the single scoring rule.

  Args:
    records: The in-scope records (dicts or attribute-style objects).
    failed_records: How failed records are counted (see module docstring).

  Returns:
    The score.
  """
  mode = parse_failed_records(failed_records)
  records = list(records)
  ok, failed = [], []
  for r in records:
    (failed if is_failed(r) else ok).append(r)
  scored = ok if mode is FailedRecords.DROP else records
  # A failed record is never a true discovery, whatever it stores.
  n_verified = sum(1 for r in ok if record_field(r, "verified"))
  n_scored = len(scored)
  counts: dict[str, int] = {}
  for r in failed:
    s = record_status(r)
    counts[s] = counts.get(s, 0) + 1
  precision = n_verified / n_scored if n_scored else None
  return Score(
      n_verified=n_verified,
      n_scored=n_scored,
      n_failed=len(failed),
      precision=precision,
      empirical_fdr=1.0 - precision if precision is not None else 0.0,
      failed_by_status=tuple(sorted(counts.items())),
      failed_records=mode,
      scored=tuple(scored),
  )


# Why a discovery round was skipped without recording an e-value:
#   target_side: the target returned no usable responses (outage, blocking).
#   short: fewer usable responses than --min_response_fraction of the
#     queries planned for the round.
#   auditor_side: the auditor did not drive the scrape or its stats reply
#     could not be parsed.
SKIP_KINDS: tuple[str, ...] = ("target_side", "short", "auditor_side")


def summarize_data_quality(
    records: Iterable[Any],
    skip_counts: Mapping[str, int] | None = None,
) -> dict[str, Any]:
  """Summarizes how much of a run's data was actually measured.

  The summary is diagnostic only: no metric reads it.

  Args:
    records: Verification records (dicts or attribute-style objects). Records
      without `n_requested` / `n_received` (older writers) add nothing to the
      response totals.
    skip_counts: Skipped discovery rounds by kind (see `SKIP_KINDS`), or None
      when the payload has no discovery loop.

  Returns:
    {n_records, n_failed, by_status, n_requested_total, n_received_total,
    response_coverage (None when nothing was requested), and, when
    `skip_counts` is given, skipped_rounds}.
  """
  records = list(records)
  by_status: dict[str, int] = {}
  n_requested = n_received = 0
  for r in records:
    s = record_status(r)
    by_status[s] = by_status.get(s, 0) + 1
    n_requested += int(record_field(r, "n_requested", 0) or 0)
    n_received += int(record_field(r, "n_received", 0) or 0)
  summary: dict[str, Any] = {
      "n_records": len(records),
      "n_failed": sum(n for s, n in by_status.items() if s != OK_STATUS),
      "by_status": dict(sorted(by_status.items())),
      "n_requested_total": n_requested,
      "n_received_total": n_received,
      "response_coverage": n_received / n_requested if n_requested else None,
  }
  if skip_counts is not None:
    summary["skipped_rounds"] = {
        k: int(skip_counts.get(k, 0)) for k in SKIP_KINDS
    }
  return summary


def log_data_quality(
    data_quality: Mapping[str, Any], log_fn: Callable[[str], None]
) -> None:
  """Logs a `summarize_data_quality` result; loud when anything was lost.

  Args:
    data_quality: The summary.
    log_fn: Logger; the banner is prefixed with "WARNING" when any record
      failed, any round was skipped or any requested response is missing.
  """
  skipped = data_quality.get("skipped_rounds") or {}
  n_skipped = sum(skipped.values())
  n_missing = data_quality.get("n_requested_total", 0) - data_quality.get(
      "n_received_total", 0
  )
  coverage = data_quality.get("response_coverage")
  parts = [
      f"{data_quality.get('n_failed', 0)}/{data_quality.get('n_records', 0)}"
      " verification record(s) failed"
  ]
  if data_quality.get("n_failed"):
    by_status = ", ".join(
        f"{s}={n}"
        for s, n in data_quality.get("by_status", {}).items()
        if s != OK_STATUS
    )
    parts[0] += f" ({by_status})"
  if coverage is not None:
    parts.append(
        f"responses {data_quality['n_received_total']}/"
        f"{data_quality['n_requested_total']} ({coverage:.1%})"
    )
  if skipped:
    parts.append(
        f"{n_skipped} discovery round(s) skipped ("
        + ", ".join(f"{k}={v}" for k, v in skipped.items())
        + ")"
    )
  noisy = data_quality.get("n_failed") or n_skipped or n_missing > 0
  prefix = "[Data Quality] WARNING: " if noisy else "[Data Quality] "
  log_fn(prefix + "; ".join(parts))
