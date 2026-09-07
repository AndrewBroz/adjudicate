from adjudicate.apply import Edit, apply_edits, match_case


def test_match_case():
    assert match_case("colour", "color") == "color"
    assert match_case("Colour", "color") == "Color"
    assert match_case("COLOUR", "color") == "COLOR"


def test_apply_multiple_on_one_line_preserves_offsets():
    text = "The Colour of colours and COLOUR.\n"
    edits = [
        Edit(1, 5, 10, "Colour", "color", "x"),
        Edit(1, 15, 21, "colours", "colors", "x"),
        Edit(1, 27, 32, "COLOUR", "color", "x"),
    ]
    out, skipped = apply_edits(text, edits)
    assert out == "The Color of colors and COLOR.\n"
    assert skipped == []


def test_apply_across_lines_with_unicode():
    text = "first — colour\nsecond centre\n"
    edits = [Edit(1, 9, 14, "colour", "color", "x"), Edit(2, 8, 13, "centre", "center", "x")]
    out, skipped = apply_edits(text, edits)
    assert out == "first — color\nsecond center\n"
    assert not skipped


def test_mismatched_original_is_skipped():
    text = "a colour b\n"
    out, skipped = apply_edits(text, [Edit(1, 3, 8, "centre", "center", "x")])
    assert out == text
    assert len(skipped) == 1


def test_overlapping_edits_keep_first():
    text = "colourful\n"
    edits = [Edit(1, 1, 9, "colourful", "colorful", "x"), Edit(1, 1, 6, "colour", "color", "x")]
    out, skipped = apply_edits(text, edits)
    assert out == "colorful\n"
    assert len(skipped) == 1
