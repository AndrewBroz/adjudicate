# adjudicate

A small library for one pattern: a detector proposes a span in a document
and a short list of candidate replacements; a local language model picks
one of them, or KEEP, from the surrounding sentence and a rule of thumb;
the applier splices the choice in with case preserved. Every change is a
bounded, logged, reasoned edit. The model never rewrites prose.

Used by [stylefix](../stylefix) (US/UK spelling and style) and
[proofix](../proofix) (typos, punctuation, commas).

Modules: `llm` (OpenAI-compatible client with `dgx`/`ollama` presets),
`core` (Item, Decision, Cache, `adjudicate`), `apply` (Edit,
`apply_edits`), `context` (sentence context, Markdown prose regions).
