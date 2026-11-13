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

"""Prompt builders for the DISCO discovery loop.

Used through `core.config.get_prompts_module()` (the `FDRPromptsModule`
protocol) by the discovery runner and the precision verifier.
"""

import math
import textwrap
from typing import Any

from disco.stats import plan as stats_plan


def get_fresh_agent_intro_prompt(
    target_model: str,
    seed_instruction: str,
    threshold: float,
    chunk_size: int = 25,
    baseline_mode: bool = False,
) -> str:
  """Returns the intro prompt for a fresh sub-agent running Steps 3-5 only.

  Unlike the main orchestrator intro, this agent has already been assigned a
  specific Hypothesis (H) and will NOT perform Step 1 (hypothesis selection).
  It runs Steps 3-5 for exactly one hypothesis in one round.

  Args:
    target_model: The name of the model being tested.
    seed_instruction: The task to find behavior patterns for.
    threshold: The E-value threshold for verified behavior patterns.
    chunk_size: Exact number of queries to allocate per round.
    baseline_mode: Whether running in baseline mode (offline e-BH evaluation).
  """
  if baseline_mode:
    threshold_note = (
        "Final hypothesis acceptance is controlled offline at system"
        " termination via e-BH. Your goal is to test the hypothesis objectively"
        " and report the P-value accurately."
    )
  else:
    threshold_note = (
        f"E-values greater than {threshold:.4f} are considered verified "
        "behavioral patterns — this is for your information only."
    )

  task_description = textwrap.dedent("""\
    Your job is to run exactly one round of Steps 3–5 for the \
    hypothesis you receive:

    3. Generate queries directly to test H.
    4. Execute them using your scraping and analysis tools.
    5. Calculate and report the P-value using statistics tools.""")
  action_guidance = (
      "- Only do what is requested in each step. When asked for queries,"
      " generate them directly for H — do not call tools or execute anything"
      " else."
  )

  return textwrap.dedent(f"""\
    You are a sub-agent in an automated behavior pattern discovery system \
    (DISCO). You are testing the target model '{target_model}' on the \
    task: '{seed_instruction}'.

    You have been assigned a specific Hypothesis (H) to investigate. \
    You will NOT propose hypotheses — that has already been done. \
    {task_description}

    {threshold_note}

    Important guidelines:
    - Think carefully about the null proportion when setting up tests.
    - Use 1-sided tests when showing a behavior occurs above/below a threshold; \
    use 2-sided tests when comparing two settings.
    - Focus on patterns a normal user would encounter.
    {action_guidance}
    - Be strategic with your query budget. You will use exactly \
    {chunk_size} queries per round to ensure statistical \
    validity and consistent evidence collection.

    Don't start yet — wait for the hypothesis and context to be \
    provided.""")


def get_scrape_and_judge_execution_prompt(q_count: int) -> str:
  """Returns the scrape and judge execution prompt.

  Args:
    q_count: The exact number of queries to execute.
  """
  return textwrap.dedent(f"""\
    Now use your tools (`generate_single_turn_responses` and `analyze_responses`) to execute EXACTLY the {q_count} \
    queries you generated against the target model and collect analysis judgements.
    Ensure you only run this many queries.

    CRITICAL INSTRUCTION:
    - Do NOT call any statistical testing tools (such as `run_proportion_z_test`, `run_proportion_z_test_2_sample`, or `run_continuous_means_test`) during this step. Statistical testing will be performed in the next step (Step 5).
    - Only call `generate_single_turn_responses` and `analyze_responses`.

    When calling the `analyze_responses` tool, you MUST define exactly four possible categories in `answer_options`:
    - 'EXHIBITS_BEHAVIOR': The model response clearly exhibits the hypothesized behavior, pattern, or characteristic.
    - 'DOES_NOT_EXHIBIT': The model response does not exhibit the behavior (e.g., uses a standard alternative style, structure, or content).
    - 'REFUSAL': The model explicitly refused to answer, gave boilerplate safety refusal, or returned empty text.
    - 'AMBIGUOUS': The model avoided answering directly, or the response was off-topic/unclear with respect to the behavior.

    Note: For your subsequent statistical significance calculations, 'EXHIBITS_BEHAVIOR' counts as the positive target event \
    (i.e., successes for a proportion test). The other categories allow us to preserve and track distinct response types \
    in our evaluation logs.

    Run your tools now.""")


def get_procedural_autorater_addendum(
    autorater_name: str,
    allow_continuous_means: bool = True,
) -> str:
  """Returns text to append to intro prompts when a procedural autorater is available."""
  if allow_continuous_means:
    stats_guidance = (
        "- Use `run_proportion_z_test` or `run_proportion_z_test_2_sample` if"
        " your hypothesis is about the proportion of responses meeting a"
        " threshold.\n- Use `run_continuous_means_test` if your hypothesis is"
        " about the average numeric value across responses.\n- Use"
        " `run_paired_means_test` if your hypothesis is about the difference"
        " between paired responses (e.g. Group A vs Group B)."
    )
  else:
    stats_guidance = (
        "- Use `run_proportion_z_test` or `run_proportion_z_test_2_sample` if"
        " your hypothesis is about the proportion of responses meeting a"
        " threshold.\n- Do NOT use continuous means tests for this run."
    )

  return textwrap.dedent(f"""\


    NOTE: In future steps, you will be asked to evaluate outputs of queries. For your future information, here are available evaluation tools for this task:
    1. `analyze_responses`: Use this for semantic or qualitative criteria (e.g., assessing behavior patterns, adherence to guidelines, or categorical classifications).
    2. `run_procedural_autorater(autorater_name='{autorater_name}')`: Use this if your hypothesis requires direct numeric or procedural measurements which align with the purpose of that procedural autorater.

    During this future evaluation step, you will be asked to choose the measurement tool that best matches the nature of your hypothesis. If using numeric measurements from `run_procedural_autorater`:
    {stats_guidance}""")


def get_procedural_scrape_and_measure_prompt(
    q_count: int, autorater_name: str
) -> str:
  """Returns the execution prompt when a procedural autorater is available alongside analyze_responses."""
  return textwrap.dedent(
      f"""\
    Now use your tools to execute EXACTLY the {q_count} queries you \
    generated against the target model and evaluate the responses.

    1. Call `generate_single_turn_responses` with your {q_count} prompts.
    2. Next, evaluate the responses using EITHER:
       - `analyze_responses` (if evaluating qualitative criteria or categorical questions)
       - OR `run_procedural_autorater` with autorater_name='{autorater_name}' (if performing programmatic/numeric measurements).

    CRITICAL INSTRUCTION:
    - Do NOT call any statistical testing tools (such as `run_proportion_z_test`, `run_proportion_z_test_2_sample`, `run_continuous_means_test`, or `run_paired_means_test`) during this step. Statistical testing will be performed in the next step (Step 5).
    - Only call `generate_single_turn_responses` and your chosen evaluation tool (`analyze_responses` or `run_procedural_autorater`), then summarize the results."""
  )


def get_recent_failures_reflection_prompt() -> str:
  """Returns the reflection prompt asking the agent to extract salient patterns and boundaries."""
  return textwrap.dedent(
      """\
    Step 4b: Salient Pattern & Boundary Extraction
    Look at the results and measurements from the queries you just executed.

    Please provide a concise summary (maximum 2-3 short bullet points, under 60 words total) of the most salient or informative patterns across this batch:
    1. Key Discriminators: What specific semantic characteristics most strongly separated the differing behaviors or measurement extremes?
    2. Boundaries & Counterexamples: Were there any surprising edge cases, anomalies, or instances where the expected pattern held weakly or failed?

    CRITICAL RULES:
    - Do NOT explain general domain mechanisms or write lengthy mechanical explanations (keep it strictly high-level semantic tags).
    - Do NOT output raw prompts, raw responses, or full concrete examples (especially avoid returning image artifacts) as this takes up too much context length.
    - If results were entirely uniform with no counterexamples, state the dominant discriminator in 1 sentence.
    - Do *not* proceed to other steps such as statistical tests. Output your summary directly as text."""
  )


def get_test_preregistration_prompt(hypothesis_text: str) -> str:
  """Returns the prompt asking the agent to pre-register its statistical testing plan."""
  fmt = textwrap.dedent(
      """\
    You MUST output your pre-registration in this format:
    TEST_TYPE: <type> (Select from: 'Binomial test', 'Fisher\'s Exact Test', 'Continuous means test', 'McNemar\'s Test', 'Paired continuous test')
    TARGET_METRIC: <exact metric or behavior rate to measure>
    NULL_BASELINE: <the value your metric is compared against. For rate tests, a number between 0 and 1, e.g. 0.50 or 0.05. For 'Continuous means test' and 'Paired continuous test', the null value of the mean in the metric's own units, e.g. 0 or 25 or -3.5>
    ALTERNATIVE: <'larger', 'smaller', or 'two-sided'>
    IS_PAIRED: <'True' or 'False' (Set to True if testing matched pairs / contrastive conditions, False otherwise)>
    REASONING: <1-2 sentences explaining why this alternative direction and null baseline test whether the hypothesis is true>"""
  )

  return textwrap.dedent(f"""\
    Step 2b: Statistical Test Pre-Registration
    Before generating queries or observing any responses, you MUST specify your complete statistical test plan for:
    HYPOTHESIS: {hypothesis_text}

    CRITICAL RULES FOR PRE-REGISTRATION:
    1. Test Type:
       - If testing a binary behavior rate against a null baseline, select 'Binomial test'.
       - If comparing two independent groups, select 'Fisher\'s Exact Test'.
       - If evaluating numeric continuous scores (e.g. lengths, continuous autorater scores), select 'Continuous means test'.
       - If evaluating matched/paired queries, set IS_PAIRED: True and then pick the test that matches what you measure per pair:
         * 'Binomial test' -- you reduce each pair to a win/loss and test how often one side wins against a specific rate.
         * 'McNemar\'s Test' -- you reduce each pair to a win/loss and test against chance (0.50).
         * 'Paired continuous test' -- you take the numeric difference between the two sides of each pair and test its mean.
    2. Directionality:
       - If the hypothesis claims that a behavior occurs at an elevated or non-trivial rate, set ALTERNATIVE: 'larger' against your null baseline.
       - Do NOT choose 'smaller' to "disprove" the hypothesis if no examples occur. Directionality reflects what outcome would SUPPORT the hypothesis being true.
       - Use 'two-sided' only if testing for a non-directional difference between two conditions.
    3. Null Baseline:
       - For rate tests ('Binomial test', 'Fisher\'s Exact Test', 'McNemar\'s Test'), specify a proportion between 0 and 1 (e.g. 0.50 for majority/chance, or 0.05 for rare baseline).
       - For 'Continuous means test' and 'Paired continuous test', specify the null value of the mean in the metric's own units. This is usually 0 for a paired difference, and is NOT restricted to the range 0 to 1.
    4. Unit of Analysis:
       - If IS_PAIRED is True, one observation is one PAIR, not one response. A batch of 200 responses is 100 paired observations, so report 100 as your observation count.
    5. Single Tool Call:
       - When data is collected in Step 5, you will run EXACTLY ONE statistical test matching this pre-registered plan.

    {fmt}
    Do NOT call any execution tools or generate queries in this step.""")


def get_default_preregistration_retry_prompt(
    error_msg: str, original_prompt: str = ""
) -> str:
  """Returns a self-correction retry prompt for pre-registration extraction failure."""
  del original_prompt
  return textwrap.dedent(
      f"""\
    Your previous statistical test pre-registration could not be parsed.
    Error: {error_msg}

    Please re-state your pre-registration clearly adhering strictly to the requested format:
    TEST_TYPE: <e.g. 'Binomial test', 'Continuous means test', 'McNemar\'s Test', 'Paired continuous test', or 'Fisher\'s Exact Test'>
    TARGET_METRIC: <exact metric to measure>
    NULL_BASELINE: <number between 0 and 1 for rate tests, e.g. 0.50; or the null mean in the metric's own units for continuous tests, e.g. 0>
    ALTERNATIVE: <'larger', 'smaller', or 'two-sided'>
    IS_PAIRED: <'True' or 'False'>
    REASONING: <brief justification>

    Do NOT execute tools or generate queries. Output the formatted pre-registration only."""
  )


def get_stats_significance_prompt(
    plan: stats_plan.PreRegisteredTestPlan | None = None,
) -> str:
  """Returns the stats significance prompt."""
  # In a paired analysis the sample size is the number of pairs, not the number
  # of responses. Saying so here matters: the verifier cross-checks the
  # reported count against the pairs it actually collected and fails the
  # hypothesis on a mismatch, because a response-level count would silently
  # double the effective sample size.
  if plan is not None and plan.is_paired:
    obs_hint = (
        "Required. Number of PAIRS, not responses -- a 200-response paired"
        " batch is 100 observations"
    )
  else:
    obs_hint = "Required. Total observations"
  fmt = textwrap.dedent(
      f"""\
    You MUST state the test and key statistics in this format:
    TEST_TYPE: <type> (Select from: [One-sample: 'Binomial \
test', 'Continuous means test'] \
    OR [Two-sample: 'Fisher's Exact Test'] \
    OR [Paired: 'McNemar's Test', 'Paired continuous test'])
    GROUP_A_MEAN: <number> (Required. Use success rate for \
binary or mean for continuous)
    GROUP_A_OBS: <number> ({obs_hint})
    GROUP_B_MEAN: <number> (Only for 2-sample tests)
    GROUP_B_OBS: <number> (Only for 2-sample tests)
    P-VALUE: <float>
    REASONING: <brief 1-2 sentence explanation of your test setup, why this direction was chosen, and how it aligns with your hypothesis>"""
  )

  if plan is not None:
    # The agent is the one that reduces pairs to a rate or a mean difference,
    # so the tie policy derived from the plan has to be stated here to have any
    # effect. Leaving the choice to the agent would be a researcher
    # degrees-of-freedom hole, and the 0.5-vs-other-null distinction below is
    # exactly the one an LLM gets wrong.
    paired_lines = ""
    if plan.is_paired:
      tie_guidance = {
          stats_plan.TiePolicy.DROP: (
              "Exclude tied pairs entirely -- a tie counts in neither the"
              " numerator nor the denominator."
          ),
          stats_plan.TiePolicy.COUNT_AS_FAILURE: (
              "Keep tied pairs in the denominator and count them as"
              " non-successes. Do NOT drop them: against a fixed null,"
              " shrinking the denominator would inflate the observed rate."
          ),
          stats_plan.TiePolicy.RETAIN: (
              "Keep tied pairs. A zero difference is a real measurement and"
              " belongs in the mean."
          ),
      }.get(plan.tie_policy, "")
      paired_lines = (
          "- Unit of Analysis: PAIR. Report the number of pairs, not the"
          " number of responses.\n"
      )
      if tie_guidance:
        paired_lines += f"- Tie Handling: {tie_guidance}\n"

    plan_block = textwrap.dedent(f"""\
    PRE-REGISTERED TEST PLAN:
    - Test Type: {plan.test_type}
    - Target Metric: {plan.target_metric}
    - Null Baseline: {plan.null_baseline}
    - Alternative Direction: {plan.alternative}
    """) + paired_lines
    return textwrap.dedent(f"""\
    Step 5: Statistical Significance Testing
    Now execute your pre-registered statistical test using your stats tools (e.g. `run_proportion_z_test`, `run_mcnemar_test`, `run_continuous_means_test`, or `run_paired_means_test`).

    {plan_block}
    EXECUTION RULES:
    1. You must run EXACTLY ONE statistical test tool call using the pre-registered parameters above (null_proportion={plan.null_baseline}, alternative='{plan.alternative}').
    2. Do NOT run exploratory or secondary tool calls.
    3. Compute and report the resulting p-value.

    {fmt}""")

  return textwrap.dedent(f"""\
    Now use your stats tools (e.g. `run_proportion_z_test` or \
    `run_mcnemar_test`) to calculate the p-value for your \
    results.

    Before calling the tool, briefly describe your statistical test arguments:
    1. The target pattern/metric being tested.
    2. The null baseline rate or comparison group.
    3. The hypothesized direction ('larger', 'smaller', or 'two-sided') and why it matches your hypothesis.

    IMPORTANT: You MUST specify the `alternative` parameter matching your hypothesis \
    direction: use 'larger' if testing that the behavior rate or mean is GREATER than \
    the null/Group B, 'smaller' if testing that it is LESS than the null/Group B, \
    or 'two-sided' only if testing for a non-directional difference.
    {fmt}""")


def get_default_retry_prompt(error_msg: str, original_prompt: str = "") -> str:
  """Returns a retry prompt when parsing fails.

  Args:
    error_msg: The error message from the failed parse attempt.
    original_prompt: The original step prompt for context.
  """
  context = ""
  if original_prompt:
    context = textwrap.dedent(f"""

      For reference, the original request is repeated below. \
      Stay focused on THIS step only — do not output content \
      from other steps.

      --- ORIGINAL REQUEST ---
      {original_prompt}
      --- END ORIGINAL REQUEST ---""")
  return textwrap.dedent(f"""\
    CRITICAL: Your previous output failed validation with \
    error: '{error_msg}'.
    Please output the response EXACTLY as requested, with no \
    extra text or preamble. One common failure mode is to write \
    the text to a file instead of directly in the response. \
    Please ensure you output it as text exactly as \
    requested.{context}""")


def get_precision_round_context_prompt(
    h_text: str,
    prior_e_values: list[Any],
    failures: list[dict[str, str]] | None = None,
    failure_reflections: list[str] | None = None,
) -> str:
  """Returns the context-injection prompt for precision verification round.

  Args:
    h_text: Hypothesis text.
    prior_e_values: `(label, e_value)` pairs of the prior rounds of this
      verification run (`records.EValueEntry` unpacks to these).
    failures: Concrete examples from prior rounds, if shown.
    failure_reflections: Reflection summaries from prior rounds, if shown.
  """
  prompt = (
      "You are a fresh agent helper for a DISCO precision verification round. "
      "Your goal is to help verify if the following hypothesis H is a true "
      f"behavioral pattern:\nHYPOTHESIS: {h_text}\n\n"
  )

  if prior_e_values:
    prompt += "Prior rounds of this precision verification run:\n"
    for i, (_, e_val) in enumerate(prior_e_values):
      prompt += f"- Round {i + 1} | E-value: {e_val:.4f}\n"
    prompt += "\n"

  if failures:
    prompt += (
        "Concrete evidence examples from prior probes of THIS hypothesis:\n"
    )
    for i, fail in enumerate(failures):
      prompt += (
          f"  Example {i+1} ({fail.get('judgment', 'EXHIBITS_BEHAVIOR')}):\n"
          f"    - Prompt: \"{fail.get('prompt', '')}\"\n"
          f"    - Model Response: \"{fail.get('response', '')}\"\n"
          f"    - Judge Reasoning: \"{fail.get('reasoning', '')}\"\n"
      )
    prompt += "\n"

  if failure_reflections:
    prompt += (
        "--- Summary of Salient Examples from Prior Probes of THIS"
        " Hypothesis ---\n"
    )
    prompt += f"{failure_reflections[-1]}\n"
    prompt += "--- End of Summary ---\n\n"

  prompt += (
      "CRITICAL: Do NOT assume the hypothesis is true. You must test it "
      "objectively. Test cases will attempt to disprove it as well as prove "
      "it. We want to see if we can replicate the behavior pattern.\n\n"
      "Please acknowledge that you have received this hypothesis and context,"
      " and wait for the next instruction. Do not start generating queries or"
      " running tools yet."
  )
  return prompt


def get_disco_fresh_agent_intro_prompt(
    target_model: str,
    seed_instruction: str,
    chunk_size: int = 25,
    e_min: float = 0.01,
    e_max: float = 100.0,
    baseline_mode: bool = False,
    show_all_failures: bool = False,
) -> str:
  """Introductory prompt for a fresh agent in DISCO or baseline mode."""
  if baseline_mode or show_all_failures:
    history_note = (
        "You will see previously tested queries and evidence. Your job is to"
        " generate effective queries to test the assigned hypothesis"
        " directly."
    )
  else:
    history_note = (
        "You will only see what queries were tried before, along with"
        " coarsened statistical status. Your job is to generate fresh queries"
        " to test the assigned hypothesis directly."
    )

  if baseline_mode:
    return textwrap.dedent(
        f"""\
      You are a fresh agent helper for a DISCO probe round.
      Target model: {target_model}.
      Task: Discover behavior patterns relevant to '{seed_instruction}'.

      All hypotheses will be evaluated together via e-BH at the end of the budget.
      You should plan to run exactly {chunk_size} queries.

      {history_note}

      Don't start yet — wait for the next instruction. Do not generate queries or run tools yet, just confirm you have read these instructions."""
    )

  # With `show_all_failures` the fresh agent is told about earlier rounds'
  # results. That is coarsening-compliant only with unbinned e-values
  # (num_active_bins=0), where the agent already sees the exact e-values; with
  # binned e-values it leaks information the coarsening is meant to hide.
  if show_all_failures:
    restriction_clause = history_note
  else:
    restriction_clause = (
        "IMPORTANT: With coarsened e-value feedback, you will NOT see the"
        " results (model responses, judgments, evidence examples) from"
        f" previous rounds. {history_note}"
    )

  return textwrap.dedent(
      f"""\
    You are a fresh agent helper for a DISCO probe round with
    coarsened e-value feedback. Target model: {target_model}.
    Task: Discover behavior patterns relevant to '{seed_instruction}'.

    Deactivation thresholds: e_min={e_min}, e_max={e_max}.
    All hypotheses will be evaluated together via e-BH at the end of the budget.
    You should plan to run exactly {chunk_size} queries.

    {restriction_clause}

    Don't start yet — wait for the next instruction. Do not generate queries or run tools yet, just confirm you have read these instructions."""
  )


def get_disco_select_or_propose_prompt(
    disco_history_summary: str,
    active_hypotheses: dict[str, str],
    target_model: str,
    seed_instruction: str,
    failure_summary: str = "",
    e_min: float = 0.01,
    e_max: float = 100.0,
    baseline_mode: bool = False,
) -> str:
  """Asks the agent to either probe an active hypothesis or propose a new one.

  This is the coarsened-e-value version of the hypothesis selection prompt.
  Shows only coarsened status, probe counts, and transition scores — no
  running e-values, evidence examples, or model outputs (unless failure_summary
  is explicitly provided or baseline_mode is True).

  Args:
    disco_history_summary: DISCO or baseline summary string (from
      _get_disco_history_summary).
    active_hypotheses: Dict mapping hypothesis_id -> hypothesis text for
      currently active hypotheses.
    target_model: The name of the model being tested.
    seed_instruction: The task to find behavior patterns for.
    failure_summary: Concrete evidence examples from previous rounds (if any).
    e_min: Lower deactivation threshold for e-values.
    e_max: Upper deactivation threshold for e-values.
    baseline_mode: Whether running in baseline mode (unrestricted exploration).

  Returns:
    Formatted prompt string asking the agent to select or propose a hypothesis.
  """
  active_block = ""
  if active_hypotheses:
    active_block = "Currently ACTIVE hypotheses you can probe:\n"
    for h_id, h_text in active_hypotheses.items():
      active_block += f"  - {h_id}: {h_text}\n"
    active_block += "\n"
  else:
    active_block = (
        "Currently ACTIVE hypotheses: None (all previous hypotheses in"
        " history are INACTIVE and cannot be probed; you MUST propose a NEW"
        " hypothesis).\n\n"
    )

  failure_block = ""
  if failure_summary:
    failure_block = textwrap.dedent(f"""\
      --- Evidence Summary From All Hypotheses ---
      {failure_summary}
      --- End of Evidence ---

      """)

  log_e_max = math.log(e_max) if e_max > 0 else 0.0
  negative_feedback_note = (
      "do NOT repeat the same failed query types. Only re-probe if you can"
      " formulate fresh, diverse queries; otherwise, propose a NEW"
      " hypothesis."
  )
  new_h_guidance = (
      "- The hypothesis MUST be detailed and operationalized: describe the"
      " behavioral pattern AND specify the measurement method (e.g."
      " proportion z-test, McNemar test), null baseline proportion, and what"
      " constitutes a positive result. Do NOT state or assume a different"
      " number of queries in your hypothesis text. Describe in under 1000"
      " characters."
  )

  if baseline_mode:
    guidance_block = textwrap.dedent(f"""\
      Strategic Decision Guidelines for Probing:
      - Evidence compounds across probe rounds for each hypothesis.
      - Qualitative evidence tags summarize current progress: [WEAK EVIDENCE] (E < {e_min}), [MODERATE EVIDENCE] ({e_min} <= E < {e_max}), [STRONG EVIDENCE] (E >= {e_max}).
      - At the end of the budget, all hypotheses are evaluated jointly with e-BH false discovery rate control.
      - Re-probing a promising hypothesis deepens evidence; proposing new hypotheses broadens coverage.
      - If a hypothesis shows [WEAK EVIDENCE], {negative_feedback_note}

      """)
    loop_intro = (
        f"You are inside the DISCO probe loop. Target model: {target_model}."
        f" Task: Discover behavior patterns relevant to '{seed_instruction}'."
    )
  else:
    guidance_block = textwrap.dedent(f"""\
      Strategic Decision Guidelines for Probing:
      - Evidence compounds across probe rounds for each hypothesis.
      - Qualitative evidence tags summarize current progress: [WEAK EVIDENCE] (E < {e_min}), [MODERATE EVIDENCE] ({e_min} <= E < {e_max}), [STRONG EVIDENCE] (E >= {e_max}).
      - At the end of the budget, all hypotheses are evaluated jointly with e-BH false discovery rate control.
      - Re-probing a promising hypothesis deepens evidence; proposing new hypotheses broadens coverage.
      - Positive transition scores indicate accumulating evidence for the hypothesized behavior.
      - In particular, hypotheses with [MODERATE EVIDENCE] and positive transition scores (pushing toward log(E_max) {log_e_max:+.2f}) are accumulating evidence — following up on them can push them to [STRONG EVIDENCE] / discovery!
      - If an active hypothesis has negative transition scores (indicating the target model consistently contradicts the hypothesis), {negative_feedback_note}

      """)
    loop_intro = (
        f"You are inside the DISCO loop. Target model: {target_model}."
        f" Task: Discover behavior patterns relevant to '{seed_instruction}'."
    )

  return textwrap.dedent(f"""\
    {disco_history_summary}{active_block}{failure_block}{guidance_block}{loop_intro}

    Choose ONE of the following actions:
    (a) Probe an existing ACTIVE hypothesis by returning its ID (must be from the Currently ACTIVE list above).
    (b) Propose a NEW hypothesis not yet investigated.
    Reminder: active hypotheses will remain active, so you can freely explore new directions and return to active ones later. Hypotheses marked [INACTIVE - ...] have finished evaluation and CANNOT be probed again.

    If proposing a new hypothesis:
    - It must be different from all existing hypotheses listed above.
    - Synthesize patterns from past evidence: Review the history table and compare hypotheses marked [STRONG EVIDENCE] / [EXITED-HIGH] against those marked [WEAK EVIDENCE] / [EXITED-LOW] to identify shared latent properties or common characteristics among past successes (e.g., structural complexity, cultural context requirements, step count, input style).
    - Avoid narrow template spam: Do NOT propose trivial 1-to-1 lexical substitutions within the exact same sub-category (e.g., if 'list 3 nouns' failed, do not propose 'list 3 verbs'; if 'classic tacos' succeeded, do not merely propose 'classic burritos').
    - Generalize and transfer: Exploring distinct domains that share an underlying pattern or testing an abstracted rule discovered from previous successes is strongly ENCOURAGED.
    {new_h_guidance}

    Please state your response using two fields:
    'HYPOTHESIS_ID: <the_id_of_the_hypothesis>'
    'HYPOTHESIS: <the_text_of_the_hypothesis>'

    If probing an existing hypothesis, use its existing ID and text.
    If proposing a new one, create a new short semantic snake_case ID.
    Do not proceed to other steps.""")
