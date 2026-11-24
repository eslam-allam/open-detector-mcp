from open_detector_mcp.styles import analyze_style, style_flags

AI_PARAGRAPH = (
    "The proposed methodology demonstrates exceptional performance across a wide "
    "range of benchmark datasets. Experimental results show that our approach "
    "achieves superior accuracy compared to existing state-of-the-art methods, "
    "while maintaining computational efficiency. The robustness of the framework "
    "is further validated through extensive ablation studies, which highlight the "
    "crucial contribution of each component. These findings underscore the "
    "importance of holistic design principles and provide a seamless pathway "
    "toward practical deployment in real-world scenarios."
)

HUMAN_PARAGRAPH = (
    "We evaluate the method on eight benchmarks. It outperforms the baselines in "
    "accuracy while using comparable compute. In ablations, the gating module "
    "accounts for most of the improvement; removing either of the other two "
    "components changes accuracy by less than 0.4 points. A small deployment on a "
    "12-node cluster ran stably under bursty load, though we did not measure tail "
    "latency under sustained contention."
)


def test_ai_paragraph_gets_actionable_issues():
    result = analyze_style(AI_PARAGRAPH)
    tags = {issue["tag"] for issue in result["issues"]}
    assert "uniform_sentence_length" in tags
    assert "buzzwords" in tags
    assert "formulaic_phrases" in tags
    assert result["metrics"]["sentence_length_cv"] < 0.25
    for issue in result["issues"]:
        assert issue["detail"] and issue["hint"]


def test_human_paragraph_avoids_the_loud_issues():
    tags = set(style_flags(HUMAN_PARAGRAPH))
    assert "heavy_transitions" not in tags
    assert "buzzwords" not in tags
    assert "formulaic_phrases" not in tags


def test_short_text_does_not_get_density_hints():
    result = analyze_style("Crucial and seamless. This is short.")
    tags = {issue["tag"] for issue in result["issues"]}
    assert "nominalization" not in tags
    assert "long_sentences" not in tags
    assert "buzzwords" in tags


def test_em_dashes_and_transitions_are_counted():
    text = (
        "The result is clear — and important — for practice. Moreover, it matters "
        "for policy; furthermore, it matters for teaching. Ultimately, we expect "
        "the effect to persist in larger samples with more diverse populations."
    )
    result = analyze_style(text)
    assert result["metrics"]["em_dashes"] == 2
    assert result["metrics"]["transition_hits"]["moreover"] == 1
    assert result["metrics"]["transitions_per_1000_words"] > 0
