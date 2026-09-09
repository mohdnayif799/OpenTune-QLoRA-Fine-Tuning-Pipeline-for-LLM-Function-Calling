"""
evaluate.py

Base-vs-fine-tuned model comparison. For structured/function-calling output
specifically, ROUGE-L is the wrong primary lens - it measures word overlap,
which is largely blind to whether the correct tool was called or whether
arguments are right. Real correctness here is: does it parse as JSON at
all, did it select the correct tool(s), and does it match the reference
exactly once normalized for JSON structure (not raw text, which would
wrongly penalize {"a":1,"b":2} vs {"b":2,"a":1} as different). ROUGE-L is
kept as a secondary signal, not the headline metric.

Same testability split as train.py:
  - All metric computation here is pure logic - genuinely tested with real
    inputs, no GPU or API needed.
  - Actually generating text from base/fine-tuned models, and actually
    calling an LLM judge API, need a GPU and a real API key respectively.
    Both are injectable (generate_fn, judge_fn) so the orchestration order
    is tested via mocks; real verification happens in Colab, same as train.py.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Callable

from rouge_score import rouge_scorer


@dataclass
class JudgeScore:
    """Parsed result of one LLM-as-judge comparison."""

    task_adherence: int  # 1-5
    format_compliance: int  # 1-5
    preferred: str  # "base", "fine_tuned", or "tie"
    reasoning: str


@dataclass
class ExampleComparison:
    """Everything computed for one held-out example."""

    instruction: str
    reference: str
    base_output: str
    finetuned_output: str
    base_rouge_l: float
    finetuned_rouge_l: float
    base_json_valid: bool
    finetuned_json_valid: bool
    # None means "couldn't be assessed" (prediction or reference wasn't
    # valid JSON) - kept distinct from False ("parsed fine, but wrong"),
    # since those are different failure modes worth telling apart.
    base_tool_correct: bool | None
    finetuned_tool_correct: bool | None
    base_exact_match: bool | None
    finetuned_exact_match: bool | None
    judge_score: JudgeScore | None


@dataclass
class ComparisonReport:
    """The full evaluation report: per-example results plus aggregates."""

    examples: list[ExampleComparison]
    avg_base_rouge_l: float
    avg_finetuned_rouge_l: float
    base_json_validity_rate: float
    finetuned_json_validity_rate: float
    base_tool_accuracy: float
    finetuned_tool_accuracy: float
    base_exact_match_rate: float
    finetuned_exact_match_rate: float
    finetuned_preferred_count: int
    base_preferred_count: int
    tie_count: int


_scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)


def compute_rouge_l(prediction: str, reference: str) -> float:
    """ROUGE-L F-measure. Secondary signal only - see module docstring for why
    this isn't the right primary metric for structured/function-calling output."""
    return _scorer.score(reference, prediction)["rougeL"].fmeasure


def compute_exact_match(prediction: str, reference: str) -> bool:
    """Case-insensitive, whitespace-normalized TEXT exact match. Fine for
    plain-text tasks; too strict for JSON (key ordering breaks it) - use
    compute_json_exact_match for structured output instead."""
    normalize = lambda s: re.sub(r"\s+", " ", s.strip().lower())
    return normalize(prediction) == normalize(reference)


def _try_parse_json(text: str):
    """
    Parse text as JSON, tolerating the common real-world artifact of a model
    wrapping valid JSON in extra text (explanation, markdown fences, etc.).
    Tries direct parsing first; falls back to extracting the first {...} or
    [...] substring. Returns None (never raises) if nothing parses - a
    metric function should degrade gracefully on one bad example, not crash
    the whole evaluation run.
    """
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    match = re.search(r"(\[.*\]|\{.*\})", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(1))
        except json.JSONDecodeError:
            return None
    return None


def compute_json_validity(prediction: str) -> bool:
    """Does the output contain parseable JSON at all? The most basic
    function-calling correctness check - nothing else matters if this fails."""
    return _try_parse_json(prediction) is not None


def _extract_tool_names(parsed_answer) -> set[str]:
    """
    Extract the set of tool/function names from a parsed function-calling
    answer. Handles a single call (dict) or multiple calls (list of dicts);
    returns an empty set for anything else rather than raising, since a
    malformed shape should count as "no correct tools identified," not crash
    the run.
    """
    if isinstance(parsed_answer, dict):
        parsed_answer = [parsed_answer]
    if not isinstance(parsed_answer, list):
        return set()
    names = set()
    for call in parsed_answer:
        if isinstance(call, dict):
            name = call.get("name") or call.get("tool") or call.get("function")
            if name:
                names.add(str(name))
    return names


def compute_tool_selection_correct(prediction: str, reference: str) -> bool | None:
    """
    Did the model call the correct SET of tools, regardless of whether the
    arguments were also right? Returns None if either side isn't valid JSON
    - "wrong tool" and "not even parseable" are different failure modes,
    worth keeping distinct rather than collapsing both into False.
    """
    pred_parsed = _try_parse_json(prediction)
    ref_parsed = _try_parse_json(reference)
    if pred_parsed is None or ref_parsed is None:
        return None
    return _extract_tool_names(pred_parsed) == _extract_tool_names(ref_parsed)


def _canonicalize_for_comparison(parsed):
    """
    Normalize a parsed JSON value so list ORDER doesn't affect equality,
    matching the same reasoning compute_tool_selection_correct already uses
    (a set, not a list comparison) - xLAM-style answers are a list of
    independent function calls with no required order, so two calls in a
    different order are still a correct match, not a mismatch. Only the
    top level is order-normalized; each individual call's own structure is
    still compared exactly.
    """
    if isinstance(parsed, list):
        return sorted(json.dumps(item, sort_keys=True) for item in parsed)
    return parsed


def compute_json_exact_match(prediction: str, reference: str) -> bool | None:
    """
    Full correctness: tool name(s) AND arguments both match exactly, once
    both sides are parsed as JSON and order-normalized at the top level -
    so key ordering, whitespace, or multi-call ordering don't cause false
    negatives the way a naive comparison would. Returns None if either side
    doesn't parse.
    """
    pred_parsed = _try_parse_json(prediction)
    ref_parsed = _try_parse_json(reference)
    if pred_parsed is None or ref_parsed is None:
        return None
    return _canonicalize_for_comparison(pred_parsed) == _canonicalize_for_comparison(ref_parsed)


def find_tool_correct_but_not_exact(examples: list[ExampleComparison]) -> list[ExampleComparison]:
    """
    Diagnostic helper: examples where the fine-tuned model picked the
    right tool(s) but the full call wasn't an exact match - i.e., an
    argument-level discrepancy. Surfaces the actual failure cases so they
    can be inspected directly, rather than guessing at failure categories
    from aggregate rates alone.
    """
    return [
        e for e in examples
        if e.finetuned_tool_correct is True and e.finetuned_exact_match is False
    ]


def build_judge_prompt(instruction: str, base_output: str, finetuned_output: str) -> str:
    """
    Construct the LLM-judge prompt comparing base vs fine-tuned output on one example.

    Asks for structured JSON output specifically so parse_judge_response can
    reliably extract scores - free-form judge responses are a common source
    of flaky evaluation pipelines.
    """
    return f"""You are evaluating two AI model responses to the same instruction.

Instruction: {instruction}

Response A: {base_output}

Response B: {finetuned_output}

Score each response 1-5 on:
- task_adherence: does it actually do what the instruction asked?
- format_compliance: is the output well-formed and appropriately structured?

Then state which response you prefer overall: "A", "B", or "tie".

Respond with ONLY valid JSON in this exact shape, nothing else:
{{"a_task_adherence": <int>, "a_format_compliance": <int>, "b_task_adherence": <int>, "b_format_compliance": <int>, "preferred": "<A|B|tie>", "reasoning": "<one sentence>"}}"""


def parse_judge_response(response_text: str) -> JudgeScore:
    """
    Parse the judge's JSON response into a JudgeScore for the fine-tuned
    model (response B). Raises ValueError with the raw response included if
    parsing fails - judge output not matching the requested format is a real
    failure mode worth surfacing clearly, not swallowing silently.
    """
    try:
        data = json.loads(response_text.strip())
        preferred_map = {"A": "base", "B": "fine_tuned", "TIE": "tie"}
        return JudgeScore(
            task_adherence=data["b_task_adherence"],
            format_compliance=data["b_format_compliance"],
            preferred=preferred_map[data["preferred"].strip().upper()],
            reasoning=data["reasoning"],
        )
    except (json.JSONDecodeError, KeyError, AttributeError) as e:
        raise ValueError(
            f"Judge response did not match expected JSON format: {e}\nRaw response: {response_text}"
        )


def run_llm_judge(
    instruction: str, base_output: str, finetuned_output: str, judge_fn: Callable[[str], str]
) -> JudgeScore:
    """Build the prompt, call the judge, parse the result. judge_fn is injectable - see tests."""
    prompt = build_judge_prompt(instruction, base_output, finetuned_output)
    response = judge_fn(prompt)
    return parse_judge_response(response)


def evaluate_models(
    eval_examples: list[dict],
    generate_fn: Callable[[str], tuple[str, str]],
    judge_fn: Callable[[str], str] | None = None,
) -> ComparisonReport:
    """
    Full evaluation orchestration: for each held-out example, generate from
    both models, score with the full metric set, and optionally run the LLM judge.

    generate_fn takes an instruction and returns (base_output, finetuned_output)
    - injectable so this can be tested without a GPU. judge_fn is optional;
    pass None to skip LLM-as-judge.

    eval_examples: list of {"instruction": str, "reference": str} dicts.
    """
    examples: list[ExampleComparison] = []

    for ex in eval_examples:
        base_out, finetuned_out = generate_fn(ex["instruction"])
        reference = ex["reference"]
        judge_score = (
            run_llm_judge(ex["instruction"], base_out, finetuned_out, judge_fn)
            if judge_fn is not None
            else None
        )
        examples.append(
            ExampleComparison(
                instruction=ex["instruction"],
                reference=reference,
                base_output=base_out,
                finetuned_output=finetuned_out,
                base_rouge_l=compute_rouge_l(base_out, reference),
                finetuned_rouge_l=compute_rouge_l(finetuned_out, reference),
                base_json_valid=compute_json_validity(base_out),
                finetuned_json_valid=compute_json_validity(finetuned_out),
                base_tool_correct=compute_tool_selection_correct(base_out, reference),
                finetuned_tool_correct=compute_tool_selection_correct(finetuned_out, reference),
                base_exact_match=compute_json_exact_match(base_out, reference),
                finetuned_exact_match=compute_json_exact_match(finetuned_out, reference),
                judge_score=judge_score,
            )
        )

    n = len(examples)
    judged = [e for e in examples if e.judge_score is not None]

    def rate(predicate) -> float:
        # None and False both count against the rate - "couldn't verify" is
        # not a pass. json_validity_rate (reported alongside) is what
        # distinguishes "wrong" from "unparseable" for the reader.
        return sum(1 for e in examples if predicate(e)) / n if n else 0.0

    return ComparisonReport(
        examples=examples,
        avg_base_rouge_l=sum(e.base_rouge_l for e in examples) / n if n else 0.0,
        avg_finetuned_rouge_l=sum(e.finetuned_rouge_l for e in examples) / n if n else 0.0,
        base_json_validity_rate=rate(lambda e: e.base_json_valid),
        finetuned_json_validity_rate=rate(lambda e: e.finetuned_json_valid),
        base_tool_accuracy=rate(lambda e: e.base_tool_correct is True),
        finetuned_tool_accuracy=rate(lambda e: e.finetuned_tool_correct is True),
        base_exact_match_rate=rate(lambda e: e.base_exact_match is True),
        finetuned_exact_match_rate=rate(lambda e: e.finetuned_exact_match is True),
        finetuned_preferred_count=sum(1 for e in judged if e.judge_score.preferred == "fine_tuned"),
        base_preferred_count=sum(1 for e in judged if e.judge_score.preferred == "base"),
        tie_count=sum(1 for e in judged if e.judge_score.preferred == "tie"),
    )
