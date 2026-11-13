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

"""Runner script for the Disco red-teaming loop."""

import asyncio
from collections.abc import Sequence

from absl import app
from absl import flags

from disco import flags_lib
from disco import runners
from disco.core import config as config_lib
from disco.core import run_config
from disco.verification import min_response_fraction_flag

DISCO_FLAGS = flags_lib.define_disco_flags()

_LOG_TO_CONSOLE = flags.DEFINE_boolean(
    "log_to_console", True, "Whether to also print logs to console."
)
_VALIDATE_STATS_TEST = flags.DEFINE_boolean(
    "validate_stats_test",
    False,
    "If True, validates that the correct statistical tool was invoked during"
    " the stats step. Disabled by default.",
)


def main(argv: Sequence[str]) -> None:
  """CLI entry point for the DISCO runner."""
  if len(argv) > 1:
    raise app.UsageError("Too many command-line arguments.")

  overrides = run_config.apply_yaml_config(
      field_names=run_config.config_field_names(config_lib.DiscoConfig)
  )
  config = config_lib.DiscoConfig.from_flags(
      DISCO_FLAGS,
      log_to_console=_LOG_TO_CONSOLE.value,
      validate_stats_test=_VALIDATE_STATS_TEST.value,
      min_response_fraction=min_response_fraction_flag.value(),
      overrides=overrides,
  )
  parsers = config.build_parsers()

  runner = runners.DiscoRunner(config, parsers)
  asyncio.run(runner.run_loop())


if __name__ == "__main__":
  app.run(main)
