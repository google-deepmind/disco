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

"""Loads the baseline category pools packaged in `baselines/data/*.jsonl`.

Each pool file holds one JSON object per line. Entries are returned as plain
dicts in file order, with the key order of each line preserved.
"""

import json
import pathlib
from typing import Any

_DATA_DIR = pathlib.Path(__file__).parent / "data"


def load_pool(name: str) -> list[dict[str, Any]]:
  """Loads the pool `baselines/data/<name>.jsonl`."""
  text = (_DATA_DIR / f"{name}.jsonl").read_text(encoding="utf-8")
  return [json.loads(line) for line in text.splitlines() if line.strip()]
