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

"""Tests for verification/scoring.

The single rule is checked on a 6-record fixture, on zero / edge cases, and
against frozen copies of the earlier sites it replaced (`_legacy_*`):
`count_as_false` must reproduce the writer that never filtered on status, and
`drop` the one that scored only OK records, wherever something was scored.
"""

import dataclasses
import itertools

from absl.testing import absltest
from absl.testing import parameterized

from disco.verification import scoring

_DROP = scoring.FailedRecords.DROP
_COUNT = scoring.FailedRecords.COUNT_AS_FALSE


@dataclasses.dataclass
class _Rec:
  hypothesis_text: str
  verified: bool
  status: str = "OK"
  sample_size: int | None = 10


def _fixture() -> list[_Rec]:
  return [
      _Rec("ok_verified", True),
      _Rec("ok_unverified", False),
      _Rec("scrape_failed", False, status="SCRAPE_FAILED", sample_size=0),
      _Rec("plan_inconsistent", False, status="PLAN_INCONSISTENT"),
      _Rec("ok_zero_sample", False, sample_size=0),
      _Rec("out_of_set", True),
  ]


_IN_SET = {
    "ok_verified",
    "ok_unverified",
    "scrape_failed",
    "plan_inconsistent",
    "ok_zero_sample",
}


def _in_set(records):
  return [r for r in records if r.hypothesis_text in _IN_SET]


# --- Frozen copies of the replaced sites -------------------------------------


def _legacy_no_status_filter(records):
  """Earlier writer without status filtering (records already in scope)."""
  n_v = sum(1 for r in records if r.verified)
  n_t = len(records)
  return n_v, n_t, (n_v / n_t if n_t else None)


def _legacy_evaluated(records):
  """Earlier precision_estimation.run_precision_check."""
  n_verified = sum(1 for r in records if r.verified)
  n_evaluated = sum(1 for r in records if r.status == "OK")
  precision = n_verified / n_evaluated if n_evaluated > 0 else None
  return n_verified, n_evaluated, precision


def _subsets() -> list[list[_Rec]]:
  recs = _fixture()
  out = [[]]
  for k in (1, 2, 3, 6):
    out.extend(list(c) for c in itertools.combinations(recs, k))
  return out


class RuleTest(absltest.TestCase):

  def test_drop(self):
    s = scoring.score(_in_set(_fixture()))
    self.assertEqual((s.n_verified, s.n_scored, s.n_failed), (1, 3, 2))
    assert s.precision is not None
    self.assertAlmostEqual(s.precision, 1 / 3)
    self.assertAlmostEqual(s.empirical_fdr, 2 / 3)
    self.assertEqual(
        s.failed_by_status, (("PLAN_INCONSISTENT", 1), ("SCRAPE_FAILED", 1))
    )
    self.assertEqual(
        [r.hypothesis_text for r in s.scored],
        ["ok_verified", "ok_unverified", "ok_zero_sample"],
    )

  def test_count_as_false(self):
    s = scoring.score(_in_set(_fixture()), failed_records="count_as_false")
    self.assertEqual((s.n_verified, s.n_scored, s.n_failed), (1, 5, 2))
    self.assertEqual(s.precision, 0.2)
    self.assertIs(s.failed_records, _COUNT)

  def test_failed_record_is_never_verified(self):
    s = scoring.score(
        [_Rec("a", True, status="SCRAPE_FAILED")], failed_records=_COUNT
    )
    self.assertEqual((s.n_verified, s.n_scored, s.precision), (0, 1, 0.0))

  def test_unknown_non_ok_status_is_failed(self):
    s = scoring.score([_Rec("a", True), _Rec("b", True, "EXECUTION_ERROR")])
    self.assertEqual((s.n_scored, s.n_failed), (1, 1))
    self.assertEqual(s.failed_by_status, (("EXECUTION_ERROR", 1),))

  def test_summary_fields(self):
    s = scoring.score(_in_set(_fixture()))
    self.assertEqual(
        s.summary_fields(),
        {
            "precision": s.precision,
            "empirical_fdr": s.empirical_fdr,
            "n_verified": 1,
            "n_scored": 3,
            "n_failed": 2,
            "n_failed_by_status": {"PLAN_INCONSISTENT": 1, "SCRAPE_FAILED": 1},
            "failed_records": "drop",
        },
    )

  def test_describe_failed(self):
    s = scoring.score(_in_set(_fixture()))
    self.assertEqual(
        s.describe_failed(),
        "2 failed record(s) (PLAN_INCONSISTENT=1, SCRAPE_FAILED=1) excluded"
        " from precision/FDR",
    )
    s = scoring.score(_in_set(_fixture()), failed_records=_COUNT)
    self.assertIn("counted as false discoveries", s.describe_failed())

  def test_parse_failed_records(self):
    self.assertIs(scoring.parse_failed_records("drop"), _DROP)
    self.assertIs(scoring.parse_failed_records(_COUNT), _COUNT)
    with self.assertRaises(ValueError):
      scoring.parse_failed_records("keep")

  def test_format_precision(self):
    self.assertEqual(scoring.format_precision(None), "n/a")
    self.assertEqual(scoring.format_precision(0.25), "0.2500")


class ShortArmsReasonTest(absltest.TestCase):

  def test_each_arm_must_reach_the_fraction(self):
    self.assertEqual(
        scoring.short_arms_reason([("A", 9, 10), ("B", 9, 10)], 0.9), ""
    )
    reason = scoring.short_arms_reason([("A", 10, 10), ("B", 6, 10)], 0.9)
    self.assertEqual(
        reason,
        "INSUFFICIENT_RESPONSES: arm 'B' 6/10 responses received (< 0.9"
        " required per arm)",
    )

  def test_pooled_count_is_not_enough(self):
    # 18/20 pooled would pass `short_batch_reason`; arm B alone is 8/10.
    self.assertNotEmpty(
        scoring.short_arms_reason([("A", 10, 10), ("B", 8, 10)], 0.9)
    )

  def test_lists_every_short_arm(self):
    reason = scoring.short_arms_reason([("A", 0, 5), ("B", 2, 5)], 0.9)
    self.assertIn("arm 'A' 0/5, arm 'B' 2/5", reason)

  def test_disabled_by_zero_fraction(self):
    self.assertEqual(
        scoring.short_arms_reason([("A", 0, 5), ("B", 1, 5)], 0.0), ""
    )


class StatusTest(parameterized.TestCase):

  @parameterized.parameters(
      ({}, "OK"),
      ({"sample_size": None}, "OK"),
      ({"sample_size": 5}, "OK"),
      ({"sample_size": 0}, "SCRAPE_FAILED"),
      ({"sample_size": -1}, "SCRAPE_FAILED"),
      # An explicit status wins over the legacy inference.
      ({"status": "OK", "sample_size": 0}, "OK"),
      ({"status": "PLAN_INCONSISTENT"}, "PLAN_INCONSISTENT"),
  )
  def test_record_status(self, record, expected):
    self.assertEqual(scoring.record_status(record), expected)
    self.assertEqual(scoring.is_failed(record), expected != "OK")

  def test_failed_statuses(self):
    for status in scoring.FAILED_STATUSES:
      self.assertTrue(scoring.is_failed({"status": status}))


class ZeroCaseTest(parameterized.TestCase):

  @parameterized.parameters(_DROP, _COUNT)
  def test_empty(self, mode):
    s = scoring.score([], failed_records=mode)
    self.assertEqual((s.n_scored, s.precision, s.empirical_fdr), (0, None, 0.0))

  def test_all_failed(self):
    recs = [_Rec("x", False, status="SCRAPE_FAILED")]
    s = scoring.score(recs)
    self.assertEqual((s.n_scored, s.precision, s.empirical_fdr), (0, None, 0.0))
    self.assertEqual(s.n_failed, 1)
    s = scoring.score(recs, failed_records=_COUNT)
    self.assertEqual((s.n_scored, s.precision, s.empirical_fdr), (1, 0.0, 1.0))


class LegacyParityTest(absltest.TestCase):
  """Where something is scored, each mode matches the site it replaced."""

  def test_modes_match_legacy_sites(self):
    for subset in _subsets():
      with self.subTest(subset=[r.hypothesis_text for r in subset]):
        legacy = _legacy_no_status_filter(subset)
        s = scoring.score(subset, failed_records=_COUNT)
        self.assertEqual((s.n_verified, s.n_scored, s.precision), legacy)
        legacy = _legacy_evaluated(subset)
        s = scoring.score(subset)
        self.assertEqual((s.n_verified, s.n_scored, s.precision), legacy)


class DataQualityTest(absltest.TestCase):

  def test_summary_counts_statuses_responses_and_skips(self):
    records = [
        {"status": "OK", "verified": True, "n_requested": 10, "n_received": 9},
        {"status": "SCRAPE_FAILED", "n_requested": 10, "n_received": 0},
        {"verified": False, "sample_size": 0},  # Legacy: unmeasured.
        _Rec("plan", False, status="PLAN_INCONSISTENT"),
    ]
    dq = scoring.summarize_data_quality(
        records, {"target_side": 2, "auditor_side": 1}
    )
    self.assertEqual(
        dq,
        {
            "n_records": 4,
            "n_failed": 3,
            "by_status": {
                "OK": 1,
                "PLAN_INCONSISTENT": 1,
                "SCRAPE_FAILED": 2,
            },
            "n_requested_total": 20,
            "n_received_total": 9,
            "response_coverage": 0.45,
            "skipped_rounds": {
                "target_side": 2,
                "short": 0,
                "auditor_side": 1,
            },
        },
    )

  def test_summary_without_requests_or_loop(self):
    dq = scoring.summarize_data_quality([{"status": "OK"}])
    self.assertIsNone(dq["response_coverage"])
    self.assertNotIn("skipped_rounds", dq)

  def test_banner_is_loud_only_when_data_was_lost(self):
    lines = []
    clean = scoring.summarize_data_quality(
        [{"status": "OK", "n_requested": 3, "n_received": 3}],
        {"target_side": 0},
    )
    scoring.log_data_quality(clean, lines.append)
    self.assertNotIn("WARNING", lines[-1])

    for lossy in (
        scoring.summarize_data_quality([{"status": "SCRAPE_FAILED"}]),
        scoring.summarize_data_quality(
            [{"status": "OK", "n_requested": 3, "n_received": 2}]
        ),
        scoring.summarize_data_quality([], {"auditor_side": 1}),
    ):
      scoring.log_data_quality(lossy, lines.append)
      self.assertIn("WARNING", lines[-1])
    self.assertIn("SCRAPE_FAILED=1", lines[1])
    self.assertIn("responses 2/3", lines[2])
    self.assertIn("auditor_side=1", lines[3])


if __name__ == "__main__":
  absltest.main()
