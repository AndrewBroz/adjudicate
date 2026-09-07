import json

from adjudicate.core import Cache, Decision, Item, adjudicate, parse_response, render_batch
from adjudicate.llm import Endpoint


def item(i=1, word="license", options=("licence", "license", "KEEP")):
    return Item(i, word, options, "rule", f"a [[{word}]] b", 3, 5, 11)


def test_parse_response_variants():
    assert parse_response('{"decisions":[{"id":1,"choice":"licence","confidence":0.9}]}')[1]["choice"] == "licence"
    assert parse_response('```json\n[{"id": "2", "choice": "KEEP"}]\n```')[2]["choice"] == "KEEP"
    assert parse_response("Sure! {\"decisions\": [{\"id\": 3, \"choice\": \"x\"}]} done") == {3: {"id": 3, "choice": "x"}}
    assert parse_response("nonsense") == {}


def test_render_batch():
    s = render_batch([item()])
    assert 'Item 1 — word: "license" — options: licence | license | KEEP' in s
    assert "Context: a [[license]] b" in s


class FakeEndpoint(Endpoint):
    def __init__(self, reply):
        super().__init__(url="http://x/v1", model="fake", name="fake")
        self.reply, self.calls = reply, 0

    def chat(self, system, user, timeout=120.0, json_mode=True):
        self.calls += 1
        return self.reply(user) if callable(self.reply) else self.reply


def test_adjudicate_validates_and_caches(tmp_path):
    cache = Cache(tmp_path / "c.json")
    ep = FakeEndpoint(json.dumps({"decisions": [
        {"id": 1, "choice": "Licence", "confidence": 0.95, "reason": "noun"},
        {"id": 2, "choice": "bogus", "confidence": 0.99},
    ]}))
    items = [item(1), item(2, "practice", ("practise", "practice", "KEEP"))]
    ds = adjudicate(ep, "uk", items, cache)
    assert ds[0].choice == "licence" and ds[0].replacement == "licence"
    assert ds[1].choice is None and "invalid" in ds[1].reason
    assert ep.calls == 1
    # Second run: item 1 is served from cache, item 2 asked again.
    ds2 = adjudicate(ep, "uk", items, Cache(tmp_path / "c.json"))
    assert ds2[0].cached and ep.calls == 2


def test_replacement_none_for_keep_and_same_word():
    assert Decision(item(), "KEEP", 1, "").replacement is None
    assert Decision(item(), "license", 1, "").replacement is None
    assert Decision(item(), "licence", 1, "").replacement == "licence"


def test_no_endpoint_marks_undecided():
    ds = adjudicate(None, "uk", [item()], Cache(None))
    assert ds[0].choice is None and "no model" in ds[0].reason


def test_batches_run_in_parallel_and_merge():
    def reply(user):
        ids = [int(l.split()[1]) for l in user.splitlines() if l.startswith("Item ")]
        return json.dumps({"decisions": [{"id": i, "choice": "KEEP", "confidence": 1} for i in ids]})
    ep = FakeEndpoint(reply)
    items = [item(i) for i in range(1, 46)]
    ds = adjudicate(ep, "uk", items, Cache(None), batch_size=20, parallel=3)
    assert ep.calls == 3 and all(d.choice == "KEEP" for d in ds)


def test_system_prompt_and_scope_key():
    from adjudicate.core import system_prompt, DEFAULT_KEEP_RULE
    s = system_prompt("Task text.", "Keep rule.")
    assert s.startswith("Task text.") and "Keep rule." in s and "decisions" in s
    assert "KEEP" in DEFAULT_KEEP_RULE
    a = Cache.key("us", item()); b = Cache.key("uk", item()); c = Cache.key("us", item(options=("x", "KEEP")))
    assert len({a, b, c}) == 3
