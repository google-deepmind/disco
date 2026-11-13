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

"""The `--failed_records` flag shared by every precision/FDR writer and reader.

Defined once here so binaries that import several readers do not define it
twice.
"""

from absl import flags

from disco.verification import scoring

FAILED_RECORDS = flags.DEFINE_enum(
    "failed_records",
    scoring.FailedRecords.DROP.value,
    list(scoring.FAILED_RECORDS_CHOICES),
    scoring.FAILED_RECORDS_HELP,
)


def value() -> scoring.FailedRecords:
  """Returns the flag's value, or its default before flags are parsed."""
  if not flags.FLAGS.is_parsed():
    return scoring.FailedRecords(FAILED_RECORDS.default)
  return scoring.FailedRecords(FAILED_RECORDS.value)
