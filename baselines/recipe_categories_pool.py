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

"""Taxonomy category pool for recipe translation discovery.

The pool is data in `data/recipe_categories.jsonl` (400 dishes: 25 categories x
16 dishes), one `{"dish_en": ..., "dish_fr": ..., "category": ...}` object per
line.
"""

from typing import Any

from disco.baselines import pool_loader

# 400-dish category pool (25 categories x 16 dishes)
RECIPE_CATEGORIES: list[dict[str, Any]] = pool_loader.load_pool(
    "recipe_categories"
)
