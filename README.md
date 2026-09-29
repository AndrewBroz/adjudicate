# adjudicate

A small library for one pattern: a detector proposes a span in a document
and a short list of candidate replacements; a local language model picks
one of them, or KEEP, from the surrounding sentence and a rule of thumb;
the applier splices the choice in with case preserved. Every change is a
bounded, logged, reasoned edit. The model never rewrites prose.

It is the toolkit behind [stylefix](https://github.com/AndrewBroz/stylefix)
(US/UK spelling and style) and [proofix](https://github.com/AndrewBroz/proofix)
(typos, punctuation, commas). It is published because they depend on it; the
API may change between 0.x releases.

```
pip install adjudicate
```

Modules: `llm` (OpenAI-compatible client and endpoint resolution), `core`
(Item, Decision, Cache, `adjudicate`), `apply` (Edit, `apply_edits`),
`context` (sentence context, Markdown prose regions), `dictionary`
(Hunspell lookups over the bundled en_US and en_GB dictionaries), `setup`
(`run_setup` and `check_endpoint`, behind the tools' `--setup` and
`--check-endpoint`).

## Choosing the model endpoint

`resolve_endpoint(app, preset="auto", url=None, model=None, key=None,
think=False, timeout=None)` merges settings from, highest first:

1. its arguments (the calling tool's `--url`, `--model`, `--timeout`)
2. `<APP>_LLM_URL`, `<APP>_LLM_MODEL`, `<APP>_LLM_KEY`, e.g. `STYLEFIX_LLM_URL`
3. `~/.config/<app>/config.toml`
4. `ADJUDICATE_LLM_URL`, `ADJUDICATE_LLM_MODEL`, `ADJUDICATE_LLM_KEY`
5. `~/.config/adjudicate/config.toml`, shared by every tool

(`$XDG_CONFIG_HOME` replaces `~/.config` when set.) Layers merge field by
field, except that a layer naming a `url` drops the `model`, key and
`thinking_switch` of the layers below it, since those described another
server. With no `url` anywhere, a local Ollama is used if it is running with
a model pulled; otherwise there is no model and callers fall back to
listing items for review.

```toml
[llm]
url = "https://api.example.com/v1"   # any OpenAI-compatible base URL
model = "some-model"                  # optional: the server's first listed model
api_key_env = "EXAMPLE_API_KEY"       # or api_key = "..."; not both
timeout = 120                         # seconds per request
thinking_switch = false               # true for vLLM/LiteLLM serving a reasoning model
```

`thinking_switch = true` sends `chat_template_kwargs: {enable_thinking:
false}`, which vLLM honours and strict APIs reject; if a server answers 400
or 422 to it, it is dropped for the rest of the run. Unknown keys, wrong
types, empty strings, a `timeout` that is not positive, and an
`api_key_env` naming an unset variable are errors (`ConfigError`), so a typo
is never silently ignored. An `api_key_env` is read only if it survives the
merge: a higher layer's `url` or key discards it first.

## License

The code is MIT (see `LICENSE`). The dictionaries in `adjudicate/dict/`
keep their own licenses: en_US under the SCOWL notice in
`LICENSE-en_US.txt`, en_GB under the LGPL as stated in `LICENSE-en_GB.txt`.
