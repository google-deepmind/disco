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

"""Prompts for the DISCO behavior pattern discovery loop.

Re-exports the builders that moved into the `prompting/` package; this module is
still the `FDRPromptsModule` returned by `core.config.get_prompts_module()`.
"""

from disco.prompting import discovery
from disco.prompting import querygen
from disco.prompting import taxonomy

get_fresh_agent_intro_prompt = discovery.get_fresh_agent_intro_prompt
get_scrape_and_judge_execution_prompt = (
    discovery.get_scrape_and_judge_execution_prompt
)
get_procedural_autorater_addendum = discovery.get_procedural_autorater_addendum
get_procedural_scrape_and_measure_prompt = (
    discovery.get_procedural_scrape_and_measure_prompt
)
get_recent_failures_reflection_prompt = (
    discovery.get_recent_failures_reflection_prompt
)
get_test_preregistration_prompt = discovery.get_test_preregistration_prompt
get_default_preregistration_retry_prompt = (
    discovery.get_default_preregistration_retry_prompt
)
get_stats_significance_prompt = discovery.get_stats_significance_prompt
get_default_retry_prompt = discovery.get_default_retry_prompt
get_precision_round_context_prompt = (
    discovery.get_precision_round_context_prompt
)
get_disco_fresh_agent_intro_prompt = (
    discovery.get_disco_fresh_agent_intro_prompt
)
get_disco_select_or_propose_prompt = (
    discovery.get_disco_select_or_propose_prompt
)

# Taxonomy-driven and isolated query generation prompts.
get_taxonomy_generation_prompt = taxonomy.get_taxonomy_generation_prompt
get_taxonomy_meta_prompt = taxonomy.get_taxonomy_meta_prompt
get_taxonomy_critic_prompt = taxonomy.get_taxonomy_critic_prompt
get_isolated_query_generation_prompt = (
    querygen.get_isolated_query_generation_prompt
)
