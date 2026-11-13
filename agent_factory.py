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

"""Agent and sender factory for DISCO."""

import os
from typing import Callable

from disco.agent import factory
from disco.agent import senders
from disco.core import config as config_lib


def build_agent_and_sender(
    config: config_lib.DiscoConfig,
    log_fn: Callable[..., None],
) -> senders.DiscoSession:
  """Factory to build the standalone agent and sender."""
  prompts_mod = config.get_prompts_module()
  agent, sender = factory.build_agent_and_sender(
      config,
      log_fn,
      prompts_mod,
      output_dir=os.path.join(config.output_dir, config.experiment_name),
  )
  return senders.DiscoSession(
      agent=agent,
      sender=sender,
      prompts_mod=prompts_mod,
  )
