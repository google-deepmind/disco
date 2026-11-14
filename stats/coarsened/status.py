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

"""Terminal status representations and strip binning for coarsened e-processes."""

import dataclasses
import numpy as np


@dataclasses.dataclass(frozen=True)
class TerminalStatus:
  """Terminal status of a hypothesis after probing with coarsened e-processes.

  Attributes:
    kind: One of "L" (exited low), "H" (exited high), "A" (active/survived).
    bin_idx: For sub-binned active hypotheses, the 1-indexed active bin in
      1..num_bins. None when kind is "L"/"H", or when using unbinned mode
      (num_active_bins=0).
  """

  kind: str  # "L", "H", "A"
  bin_idx: int | None = None

  @classmethod
  def low(cls) -> "TerminalStatus":
    return cls(kind="L")

  @classmethod
  def high(cls) -> "TerminalStatus":
    return cls(kind="H")

  @classmethod
  def active(cls, bin_idx: int | None = None) -> "TerminalStatus":
    """Active/survived. bin_idx=None for unbinned mode (num_active_bins=0)."""
    return cls(kind="A", bin_idx=bin_idx)

  @property
  def is_binned(self) -> bool:
    """Returns True if the status has an active sub-bin index."""
    return self.bin_idx is not None

  @property
  def is_active(self) -> bool:
    """Returns True if the hypothesis survived in the active strip."""
    return self.kind == "A"

  @property
  def is_exit_low(self) -> bool:
    """Returns True if the hypothesis exited below the lower strip boundary."""
    return self.kind == "L"

  @property
  def is_exit_high(self) -> bool:
    """Returns True if the hypothesis exited above the upper strip boundary."""
    return self.kind == "H"

  def __str__(self) -> str:
    if self.kind == "A" and self.bin_idx is not None:
      return f"B_{self.bin_idx}"
    return self.kind


def parse_status(
    status: str | int | tuple[str, int | None] | TerminalStatus,
) -> TerminalStatus:
  """Parses a terminal status identifier into a TerminalStatus.

  Args:
    status: TerminalStatus instance, integer bin index, or legacy string (e.g.,
      "L", "H", "A", "B_1", "bin_2", "B3").

  Returns:
    TerminalStatus instance.

  Raises:
    ValueError: If status is unrecognized.
  """
  if isinstance(status, TerminalStatus):
    return status

  if isinstance(status, tuple) and len(status) == 2:
    kind, bin_idx = status
    if kind == "L":
      return TerminalStatus.low()
    if kind == "H":
      return TerminalStatus.high()
    return TerminalStatus.active(bin_idx=bin_idx)

  if isinstance(status, (int, np.integer)):
    val = int(status)
    if val == 0:
      return TerminalStatus.low()
    if val < 0:
      raise ValueError(f"Invalid negative bin index for status: {status!r}")
    return TerminalStatus.active(bin_idx=val)

  s = str(status).strip()
  if not s:
    raise ValueError("Empty status string cannot be parsed.")

  s_upper = s.upper()
  if s_upper in ("L", "EXIT_LOW", "LOW"):
    return TerminalStatus.low()
  if s_upper in ("H", "EXIT_HIGH", "HIGH"):
    return TerminalStatus.high()
  if s_upper in ("A", "ACTIVE", "SURVIVOR", "SURVIVED"):
    return TerminalStatus.active()

  # Check prefixes: "BIN_", "B_", "BIN", "B"
  for prefix in ("BIN_", "B_", "BIN", "B"):
    if s_upper.startswith(prefix):
      rest = s_upper[len(prefix) :]
      try:
        b_val = int(rest)
        if b_val <= 0:
          raise ValueError(
              f"Bin index must be positive (got {b_val} from {status!r})"
          )
        return TerminalStatus.active(bin_idx=b_val)
      except ValueError:
        pass

  # Try parsing bare integer string (e.g. "0" -> low, "1" -> active bin 1)
  try:
    val = int(s)
    if val == 0:
      return TerminalStatus.low()
    if val > 0:
      return TerminalStatus.active(bin_idx=val)
    raise ValueError(f"Invalid negative bin index: {val}")
  except ValueError:
    pass

  raise ValueError(f"Unrecognized terminal status: {status!r}")


def compute_strip_bins(e_min: float, e_max: float, num_bins: int) -> np.ndarray:
  """Computes log-spaced bin edges for the active strip.

  Args:
    e_min: Lower strip boundary (0.0 for binary coarsening when num_bins=1).
    e_max: Upper strip boundary.
    num_bins: Number of active sub-bins in [e_min, e_max).

  Returns:
    Array of shape (num_bins + 1,) with edges from e_min to e_max.
  """
  if e_min <= 0.0:
    if num_bins <= 1:
      return np.array([0.0, e_max], dtype=float)
    raise ValueError(
        f"e_min must be > 0 when num_bins > 1 (got e_min={e_min},"
        f" num_bins={num_bins})."
    )
  return np.geomspace(e_min, e_max, num_bins + 1)


def map_to_bin_idx(
    m_vals: float | np.ndarray,
    e_min: float,
    e_max: float,
    bin_edges: np.ndarray,
    num_bins: int,
) -> int | np.ndarray:
  """Maps e-value(s) to bin index: 0=L, 1..num_bins=active, num_bins+1=H."""
  m_arr = np.asarray(m_vals, dtype=float)
  scalar_input = m_arr.ndim == 0
  m_arr = np.atleast_1d(m_arr)

  idx = np.zeros_like(m_arr, dtype=int)
  idx[m_arr < e_min] = 0
  idx[m_arr >= e_max] = num_bins + 1
  active = (m_arr >= e_min) & (m_arr < e_max)
  if np.any(active):
    b_indices = np.searchsorted(bin_edges, m_arr[active], side="right")
    idx[active] = np.clip(b_indices, 1, num_bins)

  if scalar_input:
    return int(idx[0])
  return idx


def classify_terminal_status(
    e_val: float,
    e_min: float,
    e_max: float,
    num_bins: int = 1,
    bin_edges: np.ndarray | None = None,
) -> TerminalStatus:
  """Classifies an e-value into a TerminalStatus according to strip boundaries."""
  if e_val < e_min:
    return TerminalStatus.low()
  if e_val >= e_max:
    return TerminalStatus.high()
  if num_bins > 1:
    if bin_edges is None:
      bin_edges = compute_strip_bins(e_min, e_max, num_bins)
    b_idx = int(
        np.clip(np.searchsorted(bin_edges, e_val, side="right"), 1, num_bins)
    )
    return TerminalStatus.active(bin_idx=b_idx)
  return TerminalStatus.active()
