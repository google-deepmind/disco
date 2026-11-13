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

"""Unit tests for the GenAI client factory and configuration helpers."""

from unittest import mock
from absl.testing import absltest
from google.genai import types
from disco.models import genai_client


class GenaiClientTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    genai_client.reset_cached_client()

  def tearDown(self):
    super().tearDown()
    genai_client.reset_cached_client()

  def test_http_options_configuration(self):
    opts = genai_client.get_default_http_options()
    self.assertEqual(opts.timeout, 600 * 1000)
    self.assertIsNotNone(opts.retry_options)
    self.assertEqual(opts.retry_options.attempts, 20)
    self.assertEqual(opts.retry_options.initial_delay, 5.0)
    self.assertEqual(opts.retry_options.max_delay, 120.0)
    codes = opts.retry_options.http_status_codes
    self.assertIsNotNone(codes)
    assert codes is not None
    self.assertCountEqual(
        codes,
        [408, 429, 500, 502, 503, 504],
    )

  def test_safety_settings(self):
    settings = genai_client.DEFAULT_SAFETY_SETTINGS
    self.assertLen(settings, 4)
    categories = {s.category for s in settings}
    expected_categories = {
        types.HarmCategory.HARM_CATEGORY_HATE_SPEECH,
        types.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT,
        types.HarmCategory.HARM_CATEGORY_HARASSMENT,
        types.HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT,
    }
    self.assertEqual(categories, expected_categories)
    for s in settings:
      self.assertEqual(s.threshold, types.HarmBlockThreshold.BLOCK_NONE)

  def test_get_genai_client_caching(self):
    mock_client = mock.MagicMock()
    with (
        mock.patch.object(genai_client, "get_api_key", return_value="fake-key"),
        mock.patch.object(
            genai_client.genai, "Client", return_value=mock_client
        ) as mock_genai_class,
    ):
      client1 = genai_client.get_genai_client()
      client2 = genai_client.get_genai_client()

      self.assertIs(client1, mock_client)
      self.assertIs(client2, mock_client)
      mock_genai_class.assert_called_once()
      call_kwargs = mock_genai_class.call_args.kwargs
      self.assertEqual(call_kwargs["api_key"], "fake-key")


if __name__ == "__main__":
  absltest.main()
