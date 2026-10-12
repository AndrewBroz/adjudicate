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


def test_html_entities_are_protected():
    from adjudicate.context import protected_spans
    text = "correct&#32;**bold** and &amp; and &nbsp;x\n"
    spans = [text[s:e] for s, e in protected_spans(text)]
    assert spans == ["&#32;", "&amp;", "&nbsp;"]


def test_public_regex_names():
    import adjudicate
    from adjudicate import context
    assert adjudicate.FENCE is context.FENCE is context._FENCE
    assert adjudicate.FRONT_MATTER is context.FRONT_MATTER is context._FRONT
    assert adjudicate.HTML_ENTITY is context.HTML_ENTITY is context._HTML_ENTITY
    assert context.FRONT_MATTER.match("---\ntitle: x\n---\nbody")
    assert context.HTML_ENTITY.fullmatch("&mdash;")


from adjudicate.context import protected_spans


def protected(text):
    return [text[s:e] for s, e in protected_spans(text)]


def test_obsidian_constructs_are_protected():
    text = ("See [[Some Note|the colour note]] and ![[diagram.png]].\n"
            "Tagged #draft and #area/policy here.\n"
            "Hidden %%a private colour remark%% inline.\n"
            "A paragraph with an id ^para-1\n"
            "> [!note] Colour matters\n"
            "Inline $a_colour + b$ maths.\n"
            "$$\ncolour = 1\n$$\n")
    got = protected(text)
    for piece in ["[[Some Note|the colour note]]", "![[diagram.png]]", "#draft", "#area/policy",
                  "%%a private colour remark%%", "^para-1", "> [!note]", "$a_colour + b$",
                  "$$\ncolour = 1\n$$"]:
        assert any(piece in g for g in got), piece
    # the callout's title text stays prose
    assert not any("Colour matters" in g for g in got)


def test_near_misses_stay_prose():
    text = "# Heading\nSee issue #42, the C# guide, page#anchor, and it costs $5 and $10.\n"
    got = protected(text)
    for prose in ["# Heading", "#42", "C#", "$5", "$10"]:
        assert not any(prose in g for g in got), (prose, got)


def test_multiline_obsidian_comment():
    text = "Before %%line one\nline two%% after.\n"
    assert "%%line one\nline two%%" in protected(text)
