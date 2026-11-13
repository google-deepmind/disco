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

"""Unit tests for stats.coarsened.status."""

from absl.testing import absltest
import numpy as np

from disco.stats.coarsened import status as status_lib


class StatusTest(absltest.TestCase):

  def test_terminal_status_properties(self):
    low = status_lib.TerminalStatus.low()
    self.assertTrue(low.is_exit_low)
    self.assertFalse(low.is_exit_high)
    self.assertFalse(low.is_active)
    self.assertFalse(low.is_binned)
    self.assertEqual(str(low), "L")

    high = status_lib.TerminalStatus.high()
    self.assertTrue(high.is_exit_high)
    self.assertFalse(high.is_exit_low)
    self.assertFalse(high.is_active)
    self.assertEqual(str(high), "H")

    active_unbinned = status_lib.TerminalStatus.active()
    self.assertTrue(active_unbinned.is_active)
    self.assertFalse(active_unbinned.is_binned)
    self.assertEqual(str(active_unbinned), "A")

    active_binned = status_lib.TerminalStatus.active(bin_idx=2)
    self.assertTrue(active_binned.is_active)
    self.assertTrue(active_binned.is_binned)
    self.assertEqual(str(active_binned), "B_2")

  def test_parse_status(self):
    self.assertEqual(
        status_lib.parse_status("L"), status_lib.TerminalStatus.low()
    )
    self.assertEqual(
        status_lib.parse_status("H"), status_lib.TerminalStatus.high()
    )
    self.assertEqual(
        status_lib.parse_status("A"), status_lib.TerminalStatus.active()
    )
    self.assertEqual(
        status_lib.parse_status("B_3"),
        status_lib.TerminalStatus.active(bin_idx=3),
    )
    self.assertEqual(
        status_lib.parse_status(3), status_lib.TerminalStatus.active(bin_idx=3)
    )
    self.assertEqual(
        status_lib.parse_status(("L", None)), status_lib.TerminalStatus.low()
    )
    self.assertEqual(
        status_lib.parse_status(("B", 2)),
        status_lib.TerminalStatus.active(bin_idx=2),
    )

  def test_compute_strip_bins(self):
    edges = status_lib.compute_strip_bins(0.5, 10.0, num_bins=3)
    self.assertLen(edges, 4)
    self.assertAlmostEqual(edges[0], 0.5)
    self.assertAlmostEqual(edges[-1], 10.0)

  def test_binary_coarsening_strip_bins(self):
    edges = status_lib.compute_strip_bins(0.0, 10.0, num_bins=1)
    np.testing.assert_allclose(edges, [0.0, 10.0])
    self.assertEqual(status_lib.map_to_bin_idx(0.05, 0.0, 10.0, edges, 1), 1)
    self.assertEqual(status_lib.map_to_bin_idx(5.0, 0.0, 10.0, edges, 1), 1)
    self.assertEqual(status_lib.map_to_bin_idx(12.0, 0.0, 10.0, edges, 1), 2)
    self.assertEqual(
        status_lib.classify_terminal_status(
            0.05, e_min=0.0, e_max=10.0, num_bins=1
        ),
        status_lib.TerminalStatus.active(),
    )
    with self.assertRaises(ValueError):
      status_lib.compute_strip_bins(0.0, 10.0, num_bins=2)

  def test_map_to_bin_idx(self):
    edges = status_lib.compute_strip_bins(1.0, 100.0, num_bins=2)
    # L < 1.0 -> 0
    self.assertEqual(status_lib.map_to_bin_idx(0.5, 1.0, 100.0, edges, 2), 0)
    # H >= 100.0 -> 3
    self.assertEqual(status_lib.map_to_bin_idx(150.0, 1.0, 100.0, edges, 2), 3)
    # Active -> 1 or 2
    b1 = status_lib.map_to_bin_idx(5.0, 1.0, 100.0, edges, 2)
    self.assertEqual(b1, 1)

  def test_classify_terminal_status(self):
    low = status_lib.classify_terminal_status(0.1, e_min=0.5, e_max=10.0)
    self.assertEqual(low, status_lib.TerminalStatus.low())
    high = status_lib.classify_terminal_status(20.0, e_min=0.5, e_max=10.0)
    self.assertEqual(high, status_lib.TerminalStatus.high())
    act = status_lib.classify_terminal_status(
        2.0, e_min=0.5, e_max=10.0, num_bins=1
    )
    self.assertEqual(act, status_lib.TerminalStatus.active())


if __name__ == "__main__":
  absltest.main()
