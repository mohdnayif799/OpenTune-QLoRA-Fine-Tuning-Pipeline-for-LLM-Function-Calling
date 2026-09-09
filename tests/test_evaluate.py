"""
tests/test_evaluate.py

REAL tests: compute_rouge_l (actual rouge_score library), compute_exact_match,
build_judge_prompt, parse_judge_response (real JSON parsing + real failure
handling). These prove the metric and parsing logic is correct.

MOCKED tests: evaluate_models, run_llm_judge (fake generate_fn/judge_fn).
These prove the orchestration - not that a real model's outputs would score
well, or that a real LLM API would return parseable JSON.
"""

from evaluate import (
    build_judge_prompt,
    compute_exact_match,
    compute_json_exact_match,
    compute_json_validity,
    compute_rouge_l,
    compute_tool_selection_correct,
    evaluate_models,
    parse_judge_response,
    run_llm_judge,
)


# ---------- REAL tests ----------

def test_rouge_l_identical_text_scores_near_one():
    score = compute_rouge_l("the quick brown fox", "the quick brown fox")
    assert score > 0.99
    print(f"PASS: identical text scores {score:.3f} (near 1.0)")


def test_rouge_l_unrelated_text_scores_low():
    score = compute_rouge_l("completely different words here", "the quick brown fox jumps")
    assert score < 0.3
    print(f"PASS: unrelated text scores {score:.3f} (low, as expected)")


def test_exact_match_ignores_whitespace_and_case():
    assert compute_exact_match("  Hello   World  ", "hello world") is True
    assert compute_exact_match("hello world", "goodbye world") is False
    print("PASS: exact match normalizes whitespace/case correctly, still rejects real mismatches")


def test_judge_prompt_contains_both_outputs_and_instruction():
    prompt = build_judge_prompt("Summarize this", "base summary", "finetuned summary")
    assert "Summarize this" in prompt
    assert "base summary" in prompt
    assert "finetuned summary" in prompt
    assert "JSON" in prompt
    print("PASS: judge prompt includes instruction and both model outputs")


def test_parse_valid_judge_response():
    raw = '{"a_task_adherence": 3, "a_format_compliance": 4, "b_task_adherence": 5, "b_format_compliance": 5, "preferred": "B", "reasoning": "B is more complete"}'
    score = parse_judge_response(raw)
    assert score.task_adherence == 5
    assert score.format_compliance == 5
    assert score.preferred == "fine_tuned"
    print("PASS: valid judge JSON parses into correct JudgeScore for the fine-tuned model")


def test_parse_malformed_judge_response_raises_clear_error():
    try:
        parse_judge_response("Sure! I think Response B is better because...")
        raise AssertionError("expected ValueError, none raised")
    except ValueError as e:
        assert "Raw response" in str(e)  # original text preserved for debugging
        print("PASS: malformed (non-JSON) judge response raises a clear error with the raw text attached")


# ---------- NEW function-calling metric tests ----------

def test_json_validity_direct_and_embedded():
    assert compute_json_validity('{"name": "get_weather", "arguments": {"city": "NYC"}}') is True
    assert compute_json_validity("Sure, I'll help with that!") is False
    # Model wraps valid JSON in extra text - a real generation artifact, not
    # an edge case to ignore.
    assert compute_json_validity('Here is the call: {"name": "get_weather"} done.') is True
    print("PASS: JSON validity detects both direct JSON and JSON embedded in extra text")


def test_tool_selection_correct_is_order_and_wording_independent():
    ref = '[{"name": "get_weather", "arguments": {"city": "NYC"}}]'
    same_tool_diff_args = '[{"name": "get_weather", "arguments": {"city": "LA"}}]'
    wrong_tool = '[{"name": "get_news", "arguments": {}}]'
    not_json = "I would call get_weather for this."

    assert compute_tool_selection_correct(same_tool_diff_args, ref) is True  # right tool, wrong args
    assert compute_tool_selection_correct(wrong_tool, ref) is False
    assert compute_tool_selection_correct(not_json, ref) is None  # unparseable, not "wrong"
    print("PASS: tool selection correctly distinguishes right-tool/wrong-args from wrong-tool from unparseable")


def test_json_exact_match_ignores_key_order_but_not_value_differences():
    ref = '{"name": "get_weather", "arguments": {"city": "NYC", "units": "F"}}'
    same_but_reordered = '{"arguments": {"units": "F", "city": "NYC"}, "name": "get_weather"}'
    different_value = '{"name": "get_weather", "arguments": {"city": "NYC", "units": "C"}}'

    assert compute_json_exact_match(same_but_reordered, ref) is True, (
        "key order must not affect JSON exact match - this is the whole point "
        "of using json-aware comparison instead of plain text exact match"
    )
    assert compute_json_exact_match(different_value, ref) is False
    assert compute_json_exact_match("not json", ref) is None
    print("PASS: JSON exact match ignores key ordering but still catches real value differences")


def test_json_exact_match_ignores_multi_call_order():
    """
    The specific gap found by inspecting the real code: two correct calls
    listed in a different order than the reference were previously scored
    as NOT an exact match, even though compute_tool_selection_correct
    (which uses a set) would call both tools correct. Multi-call order
    shouldn't matter any more than key order does.
    """
    ref = (
        '[{"name": "stock_get_stock_earnings_data", "arguments": {"symbol": "AMZN"}}, '
        '{"name": "options_stock", "arguments": {"symbol": "MSFT"}}]'
    )
    same_calls_different_order = (
        '[{"name": "options_stock", "arguments": {"symbol": "MSFT"}}, '
        '{"name": "stock_get_stock_earnings_data", "arguments": {"symbol": "AMZN"}}]'
    )
    genuinely_different = (
        '[{"name": "options_stock", "arguments": {"symbol": "MSFT"}}, '
        '{"name": "stock_get_stock_earnings_data", "arguments": {"symbol": "TSLA"}}]'  # wrong symbol
    )

    assert compute_json_exact_match(same_calls_different_order, ref) is True, (
        "reordered but otherwise identical multi-call answers must still count as exact"
    )
    assert compute_json_exact_match(genuinely_different, ref) is False, (
        "a genuinely different argument value must still be caught, not masked by the order fix"
    )
    print("PASS: exact match is order-independent for multi-call answers, still catches real differences")


def test_rouge_l_blind_spot_that_motivated_these_metrics():
    """
    Directly demonstrates the actual problem: a response that calls the
    WRONG tool but happens to share surface vocabulary with the reference
    can score deceptively well on ROUGE-L, while the new metrics correctly
    flag it as wrong. This is the concrete case for not treating ROUGE-L as
    the primary signal for function-calling correctness.
    """
    reference = '{"name": "get_weather", "arguments": {"city": "Paris", "units": "C"}}'
    wrong_tool_similar_words = '{"name": "get_forecast", "arguments": {"city": "Paris", "units": "C"}}'

    rouge = compute_rouge_l(wrong_tool_similar_words, reference)
    tool_correct = compute_tool_selection_correct(wrong_tool_similar_words, reference)

    assert rouge > 0.7, f"expected deceptively high ROUGE-L overlap, got {rouge:.3f}"
    assert tool_correct is False, "the tool name is actually wrong (get_forecast != get_weather)"
    print(f"PASS: demonstrated the gap - ROUGE-L={rouge:.3f} (looks good) but "
          f"tool_selection_correct={tool_correct} (actually wrong) - this is why ROUGE-L alone misleads here")


# ---------- MOCKED tests: orchestration only ----------

def test_evaluate_models_orchestration_and_aggregates():
    eval_examples = [
        {"instruction": "add 1 and 1", "reference": "2"},
        {"instruction": "add 2 and 2", "reference": "4"},
    ]

    def fake_generate(instruction: str) -> tuple[str, str]:
        # base gets it wrong, fine-tuned gets it right - deliberately, to
        # prove the aggregate math distinguishes them correctly.
        if "1" in instruction:
            return "3", "2"
        return "5", "4"

    def fake_judge(prompt: str) -> str:
        # Always prefers the fine-tuned model, deterministically.
        return '{"a_task_adherence": 1, "a_format_compliance": 3, "b_task_adherence": 5, "b_format_compliance": 5, "preferred": "B", "reasoning": "correct"}'

    report = evaluate_models(eval_examples, generate_fn=fake_generate, judge_fn=fake_judge)

    assert len(report.examples) == 2
    assert report.avg_finetuned_rouge_l > report.avg_base_rouge_l
    assert report.finetuned_preferred_count == 2
    assert report.base_preferred_count == 0
    # New fields are wired through evaluate_models, not just correct in isolation
    assert 0.0 <= report.finetuned_json_validity_rate <= 1.0
    assert 0.0 <= report.finetuned_tool_accuracy <= 1.0
    print("PASS: evaluate_models correctly orchestrates generation + judging and aggregates results")


def test_evaluate_models_without_judge_skips_judging():
    eval_examples = [{"instruction": "test", "reference": "answer"}]
    report = evaluate_models(
        eval_examples, generate_fn=lambda i: ("answer", "answer"), judge_fn=None
    )
    assert report.examples[0].judge_score is None
    assert report.finetuned_preferred_count == 0  # nothing judged, correctly zero not crashed
    print("PASS: evaluate_models works correctly with judge_fn=None (ROUGE-only mode)")


if __name__ == "__main__":
    test_rouge_l_identical_text_scores_near_one()
    test_rouge_l_unrelated_text_scores_low()
    test_exact_match_ignores_whitespace_and_case()
    test_judge_prompt_contains_both_outputs_and_instruction()
    test_parse_valid_judge_response()
    test_parse_malformed_judge_response_raises_clear_error()
    test_json_validity_direct_and_embedded()
    test_tool_selection_correct_is_order_and_wording_independent()
    test_json_exact_match_ignores_key_order_but_not_value_differences()
    test_json_exact_match_ignores_multi_call_order()
    test_rouge_l_blind_spot_that_motivated_these_metrics()
    test_evaluate_models_orchestration_and_aggregates()
    test_evaluate_models_without_judge_skips_judging()
    print("\nAll evaluate.py tests passed.")
