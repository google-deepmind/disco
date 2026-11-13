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

"""Pydantic schemas and error models for LLM distillation and parsing."""

import dataclasses
import pydantic
from disco.stats import plan


class InsufficientInformationError(Exception):
  """Model input doesn't provide essential statistical or conceptual fields."""


class ParserExhaustedError(Exception):
  """Raised when all parser distillation attempts fail.

  Unlike InsufficientInformationError (which means the source content lacked
  the needed fields), this error indicates that the distillation LLM or JSON
  parsing repeatedly produced unparseable or invalid output.  The caller should
  ask the main agent to regenerate its response — a fresh response may distill
  more cleanly.
  """


class HypothesisExtraction(pydantic.BaseModel):
  """Pydantic model for extracting Hypothesis ID and text.

  Attributes:
    insufficient_information: Set to true ONLY if the input text does NOT
      contain a valid Hypothesis ID and text.
    hypothesis_id: The extracted Hypothesis ID.
    hypothesis_text: The extracted Hypothesis text.
  """

  insufficient_information: bool = pydantic.Field(
      default=False,
      alias="INSUFFICIENT_INFORMATION",
      description=(
          "Set to true ONLY if the input text does NOT contain a valid"
          " Hypothesis ID and text."
      ),
  )
  hypothesis_id: str | None = pydantic.Field(
      default=None, alias="HYPOTHESIS_ID"
  )
  hypothesis_text: str | None = pydantic.Field(default=None, alias="HYPOTHESIS")

  model_config = pydantic.ConfigDict(populate_by_name=True)


class StatsExtraction(pydantic.BaseModel):
  """Pydantic model for extracting statistical test results.

  Attributes:
    insufficient_information: Set to true ONLY if the input text does NOT
      contain test types or results.
    test_type: The extracted name of the statistical test.
    p_value: The extracted p-value.
    group_a_mean: The extracted mean for group A.
    group_b_mean: The extracted mean for group B.
    group_a_obs: The extracted number of observations for group A.
    group_b_obs: The extracted number of observations for group B.
    null_proportion: The extracted null proportion baseline.
    alternative: The extracted hypothesized alternative direction.
    reasoning: The extracted statistical reasoning.
    sample_std: The extracted sample standard deviation.
  """

  insufficient_information: bool = pydantic.Field(
      default=False,
      alias="INSUFFICIENT_INFORMATION",
      description=(
          "Set to true ONLY if the input text does NOT contain test types or"
          " results."
      ),
  )
  test_type: str | None = pydantic.Field(default=None, alias="TEST_TYPE")
  p_value: float | None = pydantic.Field(default=None, alias="P-VALUE")
  group_a_mean: float | None = pydantic.Field(
      default=None, alias="GROUP_A_MEAN"
  )
  group_b_mean: float | None = pydantic.Field(
      default=None, alias="GROUP_B_MEAN"
  )
  group_a_obs: int | None = pydantic.Field(default=None, alias="GROUP_A_OBS")
  group_b_obs: int | None = pydantic.Field(default=None, alias="GROUP_B_OBS")
  null_proportion: float | None = pydantic.Field(
      default=None, alias="NULL_PROPORTION"
  )
  alternative: plan.AlternativeType | None = pydantic.Field(
      default=None, alias="ALTERNATIVE"
  )
  reasoning: str | None = pydantic.Field(default=None, alias="REASONING")
  sample_std: float | None = pydantic.Field(default=None, alias="SAMPLE_STD")

  model_config = pydantic.ConfigDict(populate_by_name=True)


class TestPreregistrationExtraction(pydantic.BaseModel):
  """Pydantic model for extracting pre-registered statistical test parameters.

  Attributes:
    insufficient_information: Set to true ONLY if the input text does NOT
      contain statistical test specifications.
    test_type: The name of the statistical test planned.
    target_metric: The target behavior pattern or metric to be measured.
    null_baseline: The null baseline proportion or value to test against.
    alternative: The hypothesized direction ('larger', 'smaller', or
      'two-sided').
    reasoning: Brief justification for the choice of direction and null
      baseline.
    is_paired: Whether the experiment is paired / matched pairs.
  """

  insufficient_information: bool = pydantic.Field(
      default=False,
      alias="INSUFFICIENT_INFORMATION",
      description=(
          "Set to true ONLY if the input text does NOT contain statistical"
          " test specifications."
      ),
  )
  test_type: str | None = pydantic.Field(default=None, alias="TEST_TYPE")
  target_metric: str | None = pydantic.Field(
      default=None, alias="TARGET_METRIC"
  )
  null_baseline: float | None = pydantic.Field(
      default=None, alias="NULL_BASELINE"
  )
  alternative: plan.AlternativeType | None = pydantic.Field(
      default=None, alias="ALTERNATIVE"
  )
  reasoning: str | None = pydantic.Field(default=None, alias="REASONING")
  is_paired: bool = pydantic.Field(
      default=False,
      alias="IS_PAIRED",
      description=(
          "Set to true if this pre-registration tests a paired/contrastive"
          " experiment."
      ),
  )

  model_config = pydantic.ConfigDict(populate_by_name=True)


@dataclasses.dataclass(frozen=True)
class HypothesisResult:
  """The parsed hypothesis ID and text returned by HypothesisParser."""

  h_id: str
  h_text: str
