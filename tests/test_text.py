from open_detector_mcp.text import (
    paragraphs_from_text,
    preview,
    split_document,
    split_sentences,
    word_count,
)


def test_sentence_split_handles_abbreviations_decimals_and_initials():
    text = "We used BERT et al. (2020) on 3.14M samples. Dr. Smith agreed! Is it ready? Yes."
    assert split_sentences(text) == [
        "We used BERT et al. (2020) on 3.14M samples.",
        "Dr. Smith agreed!",
        "Is it ready?",
        "Yes.",
    ]


def test_sentence_split_keeps_quotes_with_the_sentence():
    sentences = split_sentences('He said "this works." Then we left.')
    assert sentences == ['He said "this works."', "Then we left."]


def test_paragraphs_join_hard_wrapped_lines():
    text = "This is a hard-wrapped paragraph that\ncontinues on the next line.\n\nSecond paragraph."
    assert paragraphs_from_text(text) == [
        "This is a hard-wrapped paragraph that continues on the next line.",
        "Second paragraph.",
    ]


def test_paragraphs_repair_pdf_hyphenation():
    text = "We use a trans-\nformer model here."
    assert paragraphs_from_text(text) == ["We use a transformer model here."]


def test_list_items_stay_separate():
    text = "- first item with enough words\n- second item with enough words"
    assert paragraphs_from_text(text) == [
        "- first item with enough words",
        "- second item with enough words",
    ]


def test_references_are_cut_only_on_a_heading():
    text = "Body paragraph.\n\n7 References\n\n[1] Author. Paper. 2020.\n\nAppendix text."
    assert split_document(text) == ["Body paragraph."]
    assert split_document(text, exclude_references=False)[-1] == "Appendix text."


def test_a_mention_of_references_in_prose_is_not_a_cut():
    text = "We compare references in the appendix.\n\nMore body text."
    assert split_document(text) == [
        "We compare references in the appendix.",
        "More body text.",
    ]


def test_preview_and_word_count():
    text = "one two three four five six seven eight"
    assert word_count(text) == 8
    assert preview(text, 3) == "one two three …"
    assert preview("short text", 10) == "short text"
