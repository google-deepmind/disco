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

"""Unit tests for agent sender wrappers and extraction retry behavior."""

from unittest import mock

from absl.testing import absltest

from disco.agent import senders


class SendersTest(absltest.TestCase):

  def test_disco_session_dataclass(self):
    mock_agent = mock.MagicMock()
    mock_sender = mock.MagicMock()
    mock_prompts = mock.MagicMock()
    session = senders.DiscoSession(
        agent=mock_agent,
        sender=mock_sender,
        prompts_mod=mock_prompts,
    )
    self.assertIs(session.agent, mock_agent)
    self.assertIs(session.sender, mock_sender)
    self.assertIs(session.prompts_mod, mock_prompts)

  def test_base_sender_clear_history(self):
    mock_agent = mock.MagicMock()
    sender = senders.StandaloneSender(agent=mock_agent)
    sender.clear_history()
    mock_agent.clear_history.assert_called_once()


if __name__ == "__main__":
  absltest.main()
