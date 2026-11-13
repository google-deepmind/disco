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

"""The `--min_response_fraction` flag shared by DISCO and verification binaries.

Defined once here so binaries that link several modules reading it do not
define it twice.
"""

from absl import flags

from disco.verification import scoring

MIN_RESPONSE_FRACTION = flags.DEFINE_float(
    "min_response_fraction",
    scoring.DEFAULT_MIN_RESPONSE_FRACTION,
    "Minimum fraction of requested target responses a test needs to be"
    " scored. A verification test that receives fewer gets status"
    " INSUFFICIENT_RESPONSES; a DISCO discovery round that receives fewer is"
    " skipped as 'short' without charging the budget. Paired tests and rounds"
    " count complete pairs. Before that, a text scrape still short of this"
    " fraction of its responses after the 3 regular backfill waves runs up to"
    " 5 more, backing off between them, within 20 minutes of the scrape"
    " starting; a scrape with no usable response at all is treated as an"
    " outage and not topped up. 0 disables both. Visual symmetry counts"
    " scored completions (images that never arrived are excluded), and a"
    " two-arm symmetry contrast must reach the fraction in each arm.",
    lower_bound=0.0,
    upper_bound=1.0,
)


def value() -> float:
  """Returns the flag's value, or its default before flags are parsed."""
  if not flags.FLAGS.is_parsed():
    return MIN_RESPONSE_FRACTION.default
  return MIN_RESPONSE_FRACTION.value
