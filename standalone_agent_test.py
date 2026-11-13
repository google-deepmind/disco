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

"""Tests for candidate creation and caching in standalone_agent."""

from unittest import mock

from absl.testing import absltest

from disco import standalone_agent

_MODEL = "gemini-3.1-flash-image"


class MakeCandidateTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    standalone_agent._CANDIDATE_CACHE.clear()

  def test_returns_candidate(self):
    candidate = object()
    with mock.patch.object(
        standalone_agent.target_candidate,
        "make_candidate",
        return_value=candidate,
    ) as make_candidate:
      self.assertIs(standalone_agent._make_candidate(_MODEL), candidate)
    make_candidate.assert_called_once_with(_MODEL)

  def test_reraises_error(self):
    with mock.patch.object(
        standalone_agent.target_candidate,
        "make_candidate",
        side_effect=ValueError("bad address"),
    ) as make_candidate:
      with self.assertRaisesRegex(ValueError, "bad address"):
        standalone_agent._make_candidate(_MODEL)
    make_candidate.assert_called_once()


class PrewarmCandidateTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    standalone_agent._CANDIDATE_CACHE.clear()

  def test_caches_candidate_for_later_use(self):
    candidate = object()
    with mock.patch.object(
        standalone_agent, "_make_candidate", return_value=candidate
    ) as make_candidate:
      standalone_agent.prewarm_candidate(_MODEL)
      self.assertIs(
          standalone_agent._get_or_create_candidate(_MODEL), candidate
      )
    make_candidate.assert_called_once()

  def test_propagates_creation_failure(self):
    with mock.patch.object(
        standalone_agent, "_make_candidate", side_effect=ValueError("nope")
    ):
      with self.assertRaisesRegex(ValueError, "nope"):
        standalone_agent.prewarm_candidate(_MODEL)


if __name__ == "__main__":
  absltest.main()
