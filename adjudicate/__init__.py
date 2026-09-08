"""adjudicate: bounded, auditable text edits chosen by a language model.

A detector proposes a span and a short list of candidate replacements; the
model picks one (or KEEP) from the sentence and a rule of thumb; the applier
splices the choice in. Shared by stylefix and proofix.
"""

from .apply import Edit, apply_edits, line_offsets, match_case, to_offsets  # noqa: F401
from .context import (at_sentence_start, inside_quotes, map_prose, paragraph_bounds,  # noqa: F401
                      prose_regions, protected_spans, sentence_context)
from .core import (Cache, Decision, Item, adjudicate, decide_batch, parse_response,  # noqa: F401
                   render_batch, system_prompt, DEFAULT_KEEP_RULE, match_choice)
from .llm import Endpoint, LLMError, resolve_endpoint, strip_thinking  # noqa: F401
from .dictionary import known, known_anywhere  # noqa: F401

__version__ = "0.2.2"
