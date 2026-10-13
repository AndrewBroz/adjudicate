import random

import pytest

from adjudicate import Annotation, Change, apply_changes, compose, derive_changes


def test_word_change_is_one_labelled_hunk():
    before, after = "The license was granted.\n", "The licence was granted.\n"
    [c] = derive_changes(before, after, [Annotation("license", "licence", "ToUK.Variants",
                                                     "noun", "model", 0.9)])
    assert (c.start, c.end, c.before, c.after) == (4, 11, "license", "licence")
    assert (c.check, c.decided_by, c.confidence, c.reason) == ("ToUK.Variants", "model", 0.9, "noun")


def test_unlabelled_hunk_gets_default_check():
    [c] = derive_changes('say "hi"', "say “hi”", default_check="punctuation")
    assert c.check == "punctuation" and c.decided_by == "rule"


def test_inserted_comma():
    [c] = derive_changes("red, white and blue", "red, white, and blue",
                         [Annotation("", ",", "Chicago.SerialComma", decided_by="model")])
    assert (c.before, c.after, c.check) == ("", ",", "Chicago.SerialComma")


def test_crlf_round_trip():
    before = "one\r\ntwo colour\r\nthree\r\n"
    after = "one\ntwo color\nthree\n"
    assert apply_changes(before, derive_changes(before, after)) == after


def _mutate(rng, text):
    words = text.split(" ")
    for _ in range(rng.randint(0, 6)):
        i = rng.randrange(len(words))
        op = rng.choice(["swap", "del", "ins", "punct"])
        if op == "swap":
            words[i] = words[i][::-1]
        elif op == "del" and len(words) > 1:
            words.pop(i)
        elif op == "ins":
            words.insert(i, rng.choice(["new", "x,", "“q”", "\U0001F600"]))
        else:
            words[i] += rng.choice([",", ";", "—"])
    return " ".join(words)


BASE = ("Alpha beta, gamma delta.\nEpsilon zeta eta theta iota.\n\n"
        "Kappa lambda mu nu xi omicron pi.\nRho sigma tau upsilon.\n")


@pytest.mark.parametrize("seed", range(200))
def test_derive_is_faithful(seed):
    rng = random.Random(seed)
    after = _mutate(rng, BASE)
    changes = derive_changes(BASE, after)
    assert apply_changes(BASE, changes) == after
    spans = sorted((c.start, c.end) for c in changes)
    assert all(a[1] <= b[0] for a, b in zip(spans, spans[1:]))   # disjoint


def _ids(changes, prefix):
    for n, c in enumerate(changes, 1):
        c.id, c.pass_ = f"{prefix}{n}", prefix
    return changes


@pytest.mark.parametrize("seed", range(200))
def test_compose_is_faithful(seed):
    rng = random.Random(seed)
    middle = _mutate(rng, BASE)
    final = _mutate(rng, middle)
    first = _ids(derive_changes(BASE, middle), "s")
    second = _ids(derive_changes(middle, final), "p")
    allc = compose(BASE, middle, first, second)
    assert apply_changes(BASE, allc) == final
    # rejecting every second-pass change gives the middle text
    firsts_only = [c for c in allc if c.pass_ == "s"]
    assert apply_changes(BASE, firsts_only) == middle


def test_rejecting_a_dependency_drops_its_dependents():
    original = "The colour is nice\n"
    middle = "The color is nice\n"            # stylefix
    final = "The colours is nice\n"           # proofix changed the word stylefix changed
    first = _ids(derive_changes(original, middle), "s")
    second = _ids(derive_changes(middle, final), "p")
    allc = compose(original, middle, first, second)
    merged = [c for c in allc if c.depends_on]
    assert merged and merged[0].depends_on == ("s1",)
    assert apply_changes(original, allc) == final
    # a valid selection without s1 must also drop everything depending on it
    keep = [c for c in allc if c.id != "s1" and "s1" not in c.depends_on]
    assert apply_changes(original, keep) == original


def test_apply_rejects_stale_change():
    with pytest.raises(ValueError):
        apply_changes("abc", [Change(0, 1, "x", "y")])


def test_many_crlf_lines_derive_fast():
    import time
    before = "".join(f"Line {i} of some plain prose, with colour and words.\r\n" for i in range(1000))
    after = before.replace("\r\n", "\n")
    t = time.perf_counter()
    changes = derive_changes(before, after)
    assert time.perf_counter() - t < 1.0
    assert apply_changes(before, changes) == after


def test_unrelated_fix_does_not_depend_on_adjacent_insertion():
    original = "red, whte and blue"
    middle = "red, whte, and blue"           # serial comma
    final = "red, white, and blue"           # typo fix
    first = _ids(derive_changes(original, middle), "s")
    second = _ids(derive_changes(middle, final), "p")
    allc = compose(original, middle, first, second)
    p1 = [c for c in allc if c.id == "p1"][0]
    assert p1.depends_on == ()
    assert apply_changes(original, allc) == final


@pytest.mark.parametrize("before, after, word", [
    ('the colour."', "the color.”", "colour"),
    ('"colourful"', "“colorful”", "colourful"),
    ("colour's", "color’s", "colour"),
    ("colour,", "color;", "colour"),
])
def test_model_label_survives_adjacent_punctuation(before, after, word):
    new = word.replace("colour", "color")
    cs = derive_changes(before, after, [Annotation(word, new, "ToUS.Variants", decided_by="model")])
    assert apply_changes(before, cs) == after
    [c] = [c for c in cs if c.before == word]
    assert (c.check, c.decided_by) == ("ToUS.Variants", "model")


def test_annotation_labels_its_sub_hunk():
    from adjudicate import Annotation, derive_changes
    ann = [Annotation("third party", "third-party", "adjudicated", "stub", "model", 0.9)]
    cs = derive_changes("a third party risk", "a third-party risk", ann, "x")
    assert len(cs) == 1 and (cs[0].before, cs[0].after) == (" ", "-")
    assert (cs[0].check, cs[0].reason, cs[0].decided_by, cs[0].confidence) == ("adjudicated", "stub", "model", 0.9)


def test_whole_phrase_still_labels_whole():
    from adjudicate import Annotation, derive_changes
    ann = [Annotation("a lot", "many", "g", "r", "model", 0.8)]
    cs = derive_changes("a lot of cats", "many of cats", ann, "x")
    assert len(cs) == 1 and cs[0].before == "a lot" and cs[0].decided_by == "model"


def test_sub_hunk_label_not_given_to_unrelated_hunk():
    from adjudicate import Annotation, derive_changes
    ann = [Annotation("third party", "third-party", "adjudicated", "stub", "model", 0.9)]
    cs = derive_changes("x y. third party", "x z. third-party", ann, "x")
    assert [(c.before, c.after, c.decided_by) for c in cs] == [("y", "z", "rule"), (" ", "-", "model")]


def test_typo_next_to_a_doubled_word_is_two_labelled_changes():
    before = "We recieved the the new colour scheme.\n"
    after = "We received the new colour scheme.\n"
    ann = [Annotation("recieved", "received", "typos"),
           Annotation("the the", "the", "ENGLISH_WORD_REPEAT_RULE")]
    changes = derive_changes(before, after, ann, "proofix")
    assert [(c.before.strip(), c.after, c.check) for c in changes] == [
        ("recieved", "received", "typos"), ("the", "", "ENGLISH_WORD_REPEAT_RULE")]
    assert apply_changes(before, changes) == after


def test_adjacent_annotated_edits_split_whatever_the_diff_aligns():
    before = "an the the end\n"
    after = "a the end\n"
    ann = [Annotation("an", "a", "A_AN"), Annotation("the the", "the", "REPEAT")]
    changes = derive_changes(before, after, ann, "x")
    assert sorted(c.check for c in changes) == ["A_AN", "REPEAT"]
    assert apply_changes(before, changes) == after


@pytest.mark.parametrize("before, after, ann", [
    ("How do you know that that is wrong?\n", "How do you know that is wrong?\n",
     Annotation("that that", "that", "adjudicated", "a typo", "model", 0.9)),
    ("I had had enough.\n", "I had enough.\n",
     Annotation("had had", "had", "adjudicated", "a typo", "model", 0.9)),
    ("Turn it on on Monday.\n", "Turn it on Monday.\n", Annotation("on on", "on", "basics")),
])
def test_one_annotation_labels_a_deletion_the_diff_placed_at_the_other_copy(before, after, ann):
    # difflib deletes the first copy and its space; the annotation's own sub-hunk deletes
    # the space and the second copy: the same edit, so the annotation labels it.
    [c] = derive_changes(before, after, [ann], "proofix")
    assert (c.check, c.reason, c.decided_by) == (ann.check, ann.reason, ann.decided_by)
    assert apply_changes(before, [c]) == after


def test_typo_before_a_doubled_word_keeps_both_labels():
    before = "Fix teh the the bug.\n"
    after = "Fix the the bug.\n"
    ann = [Annotation("teh", "the", "typos"), Annotation("the the", "the", "basics")]
    changes = derive_changes(before, after, ann, "proofix")
    assert [(c.before.strip(), c.after, c.check) for c in changes] == [
        ("teh", "the", "typos"), ("the", "", "basics")]
    assert apply_changes(before, changes) == after


def test_single_annotation_does_not_label_a_hunk_it_does_not_reproduce():
    cs = derive_changes("say that now\n", "say this now\n", [Annotation("that that", "that", "x")], "d")
    assert [(c.before, c.after, c.check) for c in cs] == [("that", "this", "d")]
