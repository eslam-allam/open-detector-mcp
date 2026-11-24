from open_detector_mcp.analysis import classify, compare, rewrite_plan


class FakeDetector:
    """Deterministic stand-in: paragraphs containing 'ai_signal' are AI-like."""

    def score_paragraph(self, text):
        prob = 0.97 if "ai_signal" in text else 0.03
        tokens = max(1, len(text.split()))
        return prob, [{"ai_pct": round(prob * 100, 1), "tokens": tokens,
                       "preview": text[:40]}], False

    def score_texts(self, texts):
        return [0.9 if "ai_signal" in text else 0.1 for text in texts]


HUMAN = "We ran eight experiments and report the mean of three seeds in all tables."
AI = "This ai_signal paragraph presents a comprehensive framework that underscores the pivotal role of the proposed method."


def test_classify_skips_titles_and_short_fragments():
    paragraphs = [
        "A Rather Long Paper Title Without Any Terminal Punctuation",
        "Short fragment.",
        HUMAN,
        AI,
    ]
    report = classify(paragraphs, FakeDetector())
    assert report["overall"]["paragraphs_total"] == 4
    assert report["overall"]["paragraphs_scored"] == 2
    assert report["paragraphs"][0]["label"] == "Skipped"
    assert report["paragraphs"][0]["reason"] == "heading_or_fragment"
    assert report["paragraphs"][1]["label"] == "Skipped"
    assert report["overall"]["paragraphs_flagged"] == 1
    assert 0 < report["overall"]["ai_pct"] < 100


def test_classify_windows_only_inside_flagged_paragraphs():
    ai_paragraph = (
        "This ai_signal paragraph presents a comprehensive framework. It underscores "
        "the pivotal role of the proposed method across many benchmarks. Results are "
        "strong on all datasets we tested. We also report ablations in the appendix."
    )
    report = classify([ai_paragraph, HUMAN], FakeDetector(), detail="windows")
    assert "span_analysis" in report
    assert report["span_analysis"]["candidates"]
    assert all(span["paragraph_index"] == 0 for span in report["span_analysis"]["candidates"])

    only_human = classify([HUMAN], FakeDetector(), detail="windows")
    assert only_human["span_analysis"]["candidates"] == []


def test_rewrite_plan_sorts_targets_and_includes_text():
    long_ai = (
        "This ai_signal paragraph presents a comprehensive framework for retrieval. "
        "It underscores the pivotal role of the proposed method across a wide range "
        "of benchmark datasets while maintaining efficiency. Moreover, the results "
        "demonstrate robust performance in every setting we tested. These findings "
        "provide a seamless pathway toward practical deployment in real systems."
    )
    paragraphs = [HUMAN, long_ai, long_ai.replace("pivotal", "central")]
    plan = rewrite_plan(paragraphs, FakeDetector(), top_n=5, min_ai_pct=50)
    assert plan["target_count"] == 2
    assert [target["index"] for target in plan["targets"]] == [1, 2]
    assert plan["targets"][0]["text"] == paragraphs[1]
    assert plan["targets"][0]["style_issues"]
    assert plan["next_steps"]


def test_rewrite_plan_with_no_targets_explains_next_step():
    plan = rewrite_plan([HUMAN], FakeDetector(), min_ai_pct=50)
    assert plan["targets"] == []
    assert "No scored paragraph" in plan["next_steps"][0]


def test_compare_reports_improvement_and_pair_deltas():
    result = compare([AI, HUMAN], [HUMAN, HUMAN], FakeDetector())
    assert result["result"] == "IMPROVED"
    assert result["delta_ai_pct"] < 0
    assert result["paragraphs"]["compared"] == 2
    assert result["paragraphs"]["improved"] == 1
    assert result["paragraph_deltas"][0]["delta"] < 0


def test_compare_warns_on_paragraph_count_mismatch():
    result = compare([AI, HUMAN], [HUMAN], FakeDetector())
    assert any("counts differ" in warning for warning in result["warnings"])
    assert result["paragraphs"]["added"] + result["paragraphs"]["removed"] >= 1


def test_compare_handles_insufficient_text():
    result = compare(["Short."], ["Also short."], FakeDetector())
    assert result["delta_ai_pct"] is None
    assert result["result"] == "INSUFFICIENT TEXT"
    assert result["warnings"]
