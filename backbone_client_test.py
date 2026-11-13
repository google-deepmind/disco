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

"""Unit tests for backbone_client."""

import asyncio
from unittest import mock

from absl.testing import absltest
import pydantic

from disco import backbone_client


class _SampleSchema(pydantic.BaseModel):
  field_a: str
  field_b: int


_FAKE_PNG = b"\x89PNG_fake_image_data"


class PromptConversionTest(absltest.TestCase):
  """Tests for the prompt conversion helper functions."""

  def test_gemini_string_passthrough(self):
    result = backbone_client._prompt_to_gemini_contents("hello")
    self.assertEqual(result, "hello")

  def test_gemini_multimodal_parts(self):
    prompt = [
        "Describe this image:",
        backbone_client.ImagePart(data=_FAKE_PNG, mimetype="image/png"),
    ]
    result = backbone_client._prompt_to_gemini_contents(prompt)
    self.assertIsInstance(result, list)
    self.assertLen(result, 2)
    self.assertEqual(result[0].text, "Describe this image:")
    inline_data = result[1].inline_data
    self.assertIsNotNone(inline_data)
    assert inline_data is not None
    self.assertEqual(inline_data.data, _FAKE_PNG)
    self.assertEqual(inline_data.mime_type, "image/png")


class BackboneClientTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    backbone_client.reset_backbone_client()

  def tearDown(self):
    backbone_client.reset_backbone_client()
    super().tearDown()

  def test_factory_returns_gemini_by_default(self):
    client = backbone_client.get_backbone_client("")
    self.assertIsInstance(client, backbone_client.GeminiBackboneClient)

  def test_factory_caches_instance(self):
    client1 = backbone_client.get_backbone_client("")
    client2 = backbone_client.get_backbone_client("")
    self.assertIs(client1, client2)

  def test_reset_backbone_client_clears_cache(self):
    client1 = backbone_client.get_backbone_client("")
    backbone_client.reset_backbone_client()
    client2 = backbone_client.get_backbone_client("")
    self.assertIsNot(client1, client2)
    self.assertIsInstance(client2, backbone_client.GeminiBackboneClient)

  @mock.patch.object(backbone_client.genai_client, "get_genai_client")
  def test_gemini_generation_success(self, mock_get_client):
    mock_genai_client = mock.MagicMock()
    mock_get_client.return_value = mock_genai_client
    mock_response = mock.MagicMock()
    mock_response.text = '{"field_a": "test", "field_b": 99}'

    mock_generate = mock.AsyncMock(return_value=mock_response)
    mock_genai_client.aio.models.generate_content = mock_generate

    client = backbone_client.GeminiBackboneClient()
    text = asyncio.run(
        client.generate(
            model="gemini-3.8-flash",
            prompt="Test prompt",
            temperature=0.0,
            response_schema=_SampleSchema,
        )
    )

    self.assertEqual(text, '{"field_a": "test", "field_b": 99}')
    mock_generate.assert_called_once()
    _, kwargs = mock_generate.call_args
    self.assertEqual(kwargs["model"], "models/gemini-3.8-flash")
    self.assertEqual(kwargs["contents"], "Test prompt")
    self.assertEqual(kwargs["config"].temperature, 0.0)
    self.assertEqual(kwargs["config"].response_schema, _SampleSchema)
    self.assertEqual(kwargs["config"].response_mime_type, "application/json")

  @mock.patch.object(backbone_client.genai_client, "get_genai_client")
  def test_gemini_multimodal_prompt(self, mock_get_client):
    mock_genai_client = mock.MagicMock()
    mock_get_client.return_value = mock_genai_client
    mock_response = mock.MagicMock()
    mock_response.text = "A landscape."

    mock_generate = mock.AsyncMock(return_value=mock_response)
    mock_genai_client.aio.models.generate_content = mock_generate

    client = backbone_client.GeminiBackboneClient()
    prompt = [
        "Describe this image:",
        backbone_client.ImagePart(data=_FAKE_PNG, mimetype="image/png"),
    ]
    text = asyncio.run(client.generate(model="gemini-3.8-flash", prompt=prompt))

    self.assertEqual(text, "A landscape.")
    mock_generate.assert_called_once()
    _, kwargs = mock_generate.call_args
    contents = kwargs["contents"]
    self.assertIsInstance(contents, list)
    self.assertLen(contents, 2)
    self.assertEqual(contents[0].text, "Describe this image:")
    inline_data = contents[1].inline_data
    self.assertIsNotNone(inline_data)
    assert inline_data is not None
    self.assertEqual(inline_data.data, _FAKE_PNG)
    self.assertEqual(inline_data.mime_type, "image/png")

  @mock.patch.object(backbone_client.genai_client, "get_genai_client")
  def test_gemini_empty_response_raises_error(self, mock_get_client):
    mock_genai_client = mock.MagicMock()
    mock_get_client.return_value = mock_genai_client
    mock_response = mock.MagicMock()
    mock_response.text = ""

    mock_generate = mock.AsyncMock(return_value=mock_response)
    mock_genai_client.aio.models.generate_content = mock_generate

    client = backbone_client.GeminiBackboneClient()
    with self.assertRaises(ValueError):
      asyncio.run(client.generate("gemini-3.8-flash", "Test prompt"))


if __name__ == "__main__":
  absltest.main()
