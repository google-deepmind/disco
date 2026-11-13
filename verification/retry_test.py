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

"""Unit tests for failed-record selection, merging, and backup naming."""

import datetime

from absl.testing import absltest

from disco.verification import retry


def _rec(label, status="OK", text=None, verified=False, **extra):
  return {
      "label": label,
      "hypothesis_text": text if text is not None else f"text {label}",
      "status": status,
      "verified": verified,
      **extra,
  }


class RetryTest(absltest.TestCase):

  def test_select_uses_inferred_legacy_status(self):
    records = [
        _rec("a"),
        _rec("b", "SCRAPE_FAILED"),
        _rec("c", "INSUFFICIENT_RESPONSES"),
        _rec("d", "PLAN_INCONSISTENT"),
        # Legacy: no status and nothing measured counts as SCRAPE_FAILED.
        {"label": "e", "hypothesis_text": "t", "sample_size": 0},
    ]
    self.assertEqual(
        retry.select(records, retry.DEFAULT_RETRY_STATUSES), [1, 2, 4]
    )

  def test_merge_by_label_then_text_and_counts_attempts(self):
    records = [
        _rec("a"),
        _rec("b", "SCRAPE_FAILED", n_attempts=2),
        _rec("dup", "SCRAPE_FAILED", text="Same  text"),
        _rec("c", "INSUFFICIENT_RESPONSES"),
    ]
    new = [
        _rec("c", "OK", verified=True),
        # Label changed; matched by normalized text.
        _rec("renamed", "OK", text="Same text"),
        _rec("b", "SCRAPE_FAILED"),
        # Matches nothing that was retried: dropped.
        _rec("a", "OK", verified=True),
    ]
    merged, changes = retry.merge(records, [1, 2, 3], new)
    self.assertEqual(merged[0], records[0])
    self.assertEqual(merged[1]["n_attempts"], 3)
    self.assertEqual(merged[2]["label"], "renamed")
    self.assertEqual(merged[2]["n_attempts"], 2)
    self.assertTrue(merged[3]["verified"])
    self.assertEqual(merged[3]["n_attempts"], 2)
    self.assertEqual(
        changes,
        [
            {
                "label": "c",
                "status_before": "INSUFFICIENT_RESPONSES",
                "status_after": "OK",
            },
            {
                "label": "dup",
                "status_before": "SCRAPE_FAILED",
                "status_after": "OK",
            },
            {
                "label": "b",
                "status_before": "SCRAPE_FAILED",
                "status_after": "SCRAPE_FAILED",
            },
        ],
    )
    # The inputs are not modified.
    self.assertNotIn("n_attempts", records[3])

  def test_backup_path_picks_first_free_name(self):
    existing = {"/d/r.pre_retry.json", "/d/r.pre_retry.2.json"}
    self.assertEqual(
        retry.backup_path("/d/r.json", lambda p: False), "/d/r.pre_retry.json"
    )
    self.assertEqual(
        retry.backup_path("/d/r.json", existing.__contains__),
        "/d/r.pre_retry.3.json",
    )

  def test_summary_counts(self):
    records = [
        _rec("a", verified=True),
        _rec("b"),
        _rec("c", "SCRAPE_FAILED"),
    ]
    self.assertEqual(
        retry.summary_counts(records),
        {
            "n_total": 3,
            "n_evaluated": 2,
            "n_statistical_failures": 1,
            "n_execution_errors": 1,
            "n_verification_failures": 1,
        },
    )

  def test_history_entry(self):
    changes = [
        {"label": "a", "status_before": "SCRAPE_FAILED", "status_after": "OK"},
        {
            "label": "b",
            "status_before": "SCRAPE_FAILED",
            "status_after": "SCRAPE_FAILED",
        },
    ]
    entry = retry.history_entry(
        ["SCRAPE_FAILED"], changes, now=datetime.datetime(2026, 1, 2, 3, 4, 5)
    )
    self.assertEqual(
        entry,
        {
            "timestamp": "2026-01-02T03:04:05",
            "retry_statuses": ["SCRAPE_FAILED"],
            "n_retried": 2,
            "n_recovered": 1,
            "records": changes,
        },
    )


if __name__ == "__main__":
  absltest.main()
