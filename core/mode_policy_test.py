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

"""Unit tests for mode policy selection, DISCO transitions, and baseline exits."""

from absl.testing import absltest

from disco.core import checkpoint
from disco.core import config as config_lib
from disco.core import mode_policy


def _config(**kw) -> config_lib.DiscoConfig:
  kwargs = dict(
      target_model="m",
      seed_instruction="s",
      budget=40,
      alpha=0.1,
      kappa=0.5,
      chunk_size=10,
      n_sim=200,
  )
  kwargs.update(kw)
  return config_lib.DiscoConfig(**kwargs)


def _state(g_id: str, probes: int = 1) -> checkpoint.RunState:
  state = checkpoint.RunState(remaining_budget=40)
  state.active_hypotheses.add(g_id)
  state.running_e_values[g_id] = 1.0
  state.probe_counts[g_id] = probes
  state.transition_scores[g_id] = []
  state.bin_trajectories[g_id] = []
  return state


class MakePolicyTest(absltest.TestCase):

  def test_mode_selection(self):
    self.assertIsInstance(
        mode_policy.make_policy(_config()), mode_policy.DiscoPolicy
    )
    self.assertIsInstance(
        mode_policy.make_policy(_config(method="ebh_baseline")),
        mode_policy.BaselinePolicy,
    )
    self.assertIsInstance(
        mode_policy.make_policy(_config(method="type1_baseline")),
        mode_policy.Type1BaselinePolicy,
    )

  def test_baselines_have_no_null_tables(self):
    policy = mode_policy.make_policy(_config(method="ebh_baseline"))
    self.assertTrue(policy.is_baseline)
    self.assertIsNone(policy.null_tables)
    self.assertEmpty(policy.bin_edges)
    self.assertEqual(policy.init_bin, 1)

  def test_disco_precomputes_tables(self):
    policy = mode_policy.make_policy(_config(num_active_bins=1))
    self.assertFalse(policy.is_baseline)
    self.assertIsNotNone(policy.null_tables)
    self.assertEqual(policy.effective_bins, 1)
    self.assertFalse(policy.unbinned_e_values)

  def test_disco_rejects_show_all_failures_with_positive_bins(self):
    with self.assertRaisesRegex(
        ValueError, "show_all_failures=True is not permitted"
    ):
      mode_policy.make_policy(
          _config(num_active_bins=3, show_all_failures=True)
      )


class AfterEValueTest(absltest.TestCase):

  def test_type1_freezes_at_one_over_alpha(self):
    policy = mode_policy.make_policy(_config(method="type1_baseline"))
    state = _state("h")
    policy.after_e_value(state, "h", 10.0, 0.01, 10.0, lambda *a, **k: None)
    self.assertEqual(state.terminal_statuses["h"], "H")
    self.assertEqual(state.final_e_values["h"], 10.0)
    self.assertNotIn("h", state.active_hypotheses)

  def test_type1_below_threshold_stays_active(self):
    policy = mode_policy.make_policy(_config(method="type1_baseline"))
    state = _state("h")
    policy.after_e_value(state, "h", 9.0, 0.02, 9.0, lambda *a, **k: None)
    self.assertIn("h", state.active_hypotheses)
    self.assertNotIn("h", state.terminal_statuses)

  def test_baseline_exit_low_only_when_enforced(self):
    for enforce, active in ((False, True), (True, False)):
      with self.subTest(enforce=enforce):
        policy = mode_policy.make_policy(
            _config(method="ebh_baseline", baseline_enforce_exit_low=enforce)
        )
        state = _state("h")
        low = policy.config.e_min / 2
        policy.after_e_value(state, "h", low, 0.9, low, lambda *a, **k: None)
        self.assertEqual("h" in state.active_hypotheses, active)
        if not active:
          self.assertEqual(state.terminal_statuses["h"], "L")

  def test_baseline_exit_high_threshold_is_k_bar_over_alpha(self):
    # budget=40, chunk_size=10 -> K_bar=4; alpha=0.1 -> 40.
    policy = mode_policy.BaselinePolicy(
        _config(method="ebh_baseline", baseline_exit_high=True)
    )
    self.assertIsInstance(
        mode_policy.make_policy(policy.config), mode_policy.BaselinePolicy
    )
    self.assertEqual(policy.k_bar(), 4)
    threshold = policy.exit_high_threshold()
    self.assertIsNotNone(threshold)
    self.assertAlmostEqual(threshold or 0.0, 40.0)
    # K_bar follows budget // chunk_size.
    policy = mode_policy.BaselinePolicy(
        _config(method="ebh_baseline", baseline_exit_high=True, budget=200)
    )
    self.assertEqual(policy.k_bar(), 20)
    self.assertAlmostEqual(policy.exit_high_threshold() or 0.0, 200.0)
    # Disabled by default.
    policy = mode_policy.BaselinePolicy(_config(method="ebh_baseline"))
    self.assertIsNone(policy.exit_high_threshold())

  def test_baseline_exit_high_freezes_at_threshold(self):
    policy = mode_policy.BaselinePolicy(
        _config(method="ebh_baseline", baseline_exit_high=True)
    )
    threshold = policy.exit_high_threshold() or 0.0
    self.assertGreater(threshold, 0.0)
    logs = []
    # Just below: stays active.
    state = _state("h")
    policy.after_e_value(
        state, "h", threshold * 0.99, 0.01, threshold * 0.99, logs.append
    )
    self.assertIn("h", state.active_hypotheses)
    self.assertNotIn("h", state.terminal_statuses)
    # At threshold: frozen as H with its final e-value recorded.
    state = _state("h", probes=2)
    policy.after_e_value(state, "h", threshold, 1e-3, threshold, logs.append)
    self.assertEqual(state.terminal_statuses["h"], "H")
    self.assertEqual(state.final_e_values["h"], threshold)
    self.assertNotIn("h", state.active_hypotheses)
    self.assertNotIn("h", state.running_e_values)
    self.assertTrue(any("[Baseline Exit-High]" in m for m in logs))

  def test_baseline_without_exit_high_never_freezes(self):
    policy = mode_policy.make_policy(_config(method="ebh_baseline"))
    state = _state("h")
    huge = 1e9
    policy.after_e_value(state, "h", huge, 1e-9, huge, lambda *a, **k: None)
    self.assertIn("h", state.active_hypotheses)
    self.assertNotIn("h", state.terminal_statuses)

  def test_baseline_exit_high_log_config(self):
    for enabled in (False, True):
      with self.subTest(enabled=enabled):
        policy = mode_policy.make_policy(
            _config(method="ebh_baseline", baseline_exit_high=enabled)
        )
        logs = []
        policy.log_config(logs.append)
        self.assertTrue(
            any(f"Baseline exit_high: {enabled}" in m for m in logs), logs
        )

  def test_disco_deactivates_on_exit_high(self):
    policy = mode_policy.make_policy(_config(num_active_bins=0))
    state = _state("h")
    high = policy.config.e_max * 2
    logs = []
    policy.after_e_value(state, "h", high, 1e-4, high, logs.append)
    self.assertEqual(state.terminal_statuses["h"], "H")
    self.assertNotIn("h", state.active_hypotheses)
    self.assertLen(state.transition_scores["h"], 1)
    self.assertTrue(any("Running E (private)" in m for m in logs))

  def test_disco_in_strip_stays_active(self):
    policy = mode_policy.make_policy(_config(num_active_bins=0))
    state = _state("h")
    policy.after_e_value(state, "h", 1.5, 0.3, 1.5, lambda *a, **k: None)
    self.assertIn("h", state.active_hypotheses)
    self.assertEqual(state.terminal_statuses["h"], "A")


if __name__ == "__main__":
  absltest.main()
