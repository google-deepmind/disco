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

"""System instruction for the structured-output parser model."""

import textwrap


def get_parser_system_instruction() -> str:
  """Returns the system instruction for the distillation agent."""
  return textwrap.dedent("""\
    Analyze the provided text. If the input string does NOT explicitly state the parameters specified by the \
    output constraints schema (e.g. missing targeted calculation labels or observation expectations), then \
    please set the 'INSUFFICIENT_INFORMATION' parameter to true. If you set 'INSUFFICIENT_INFORMATION' to true, \
    do not fill in any other fields.""")
