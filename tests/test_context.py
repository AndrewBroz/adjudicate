from adjudicate.context import at_sentence_start, inside_quotes, sentence_context


def test_sentence_context_marks_word():
    text = "Intro para.\n\nFirst one. The Labour Party won. Last one. Extra.\n"
    pos = text.index("Labour")
    ctx = sentence_context(text, pos, pos + 6)
    assert ctx == "First one. The [[Labour]] Party won. Last one."


def test_sentence_start_detection():
    text = "# Colour\n\n- Colour x\n\nColour a. Colour b, the Colour c: Colour d\n"
    assert at_sentence_start(text, text.index("Colour"))
    assert at_sentence_start(text, text.index("Colour", 10))
    para = text.index("Colour a")
    assert at_sentence_start(text, para)
    assert at_sentence_start(text, text.index("Colour b"))
    assert not at_sentence_start(text, text.index("Colour c"))
    assert at_sentence_start(text, text.index("Colour d"))


def test_inside_quotes():
    text = 'He said "the colour is off" and colour.\n'
    assert inside_quotes(text, text.index("colour"))
    assert not inside_quotes(text, text.rindex("colour"))
    text2 = "He said “the colour is off” and colour.\n"
    assert inside_quotes(text2, text2.index("colour"))
    assert not inside_quotes(text2, text2.rindex("colour"))


def test_inside_quotes_opening_quote_right_before_word():
    text = "we favour 'color' here.\n"
    assert inside_quotes(text, text.index("color"))


def test_link_text_and_parentheses_are_not_sentence_starts():
    text = "See the [Center for AI](https://x.y) and (Centre for Z) now.\n"
    assert not at_sentence_start(text, text.index("Center"))
    assert not at_sentence_start(text, text.index("Centre"))
