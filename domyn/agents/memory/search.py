"""Searching memories: BM25 over character trigrams, computed at every search.

A memory is a short fact, often in English (the extraction prompt asks for it), but
one written by hand or remembered by the agent can be in any language, and so can the
question that looks for it. So:

* **Text is folded** first: case, accents and compatibility forms (``É`` → ``e``,
  ``ﬁ`` → ``fi``), with every script kept — a Cyrillic or Greek word is a word.
* **Words are compared by character trigrams**, the edges of a word marked
  (``_it``, ``ita``, ``tal``, ``aly``, ``ly_`` for ``italy``), so another form of a
  word still matches: ``Italian`` finds ``Italy``, ``prefer`` finds ``preferences``.
  A script written without spaces is one long word and is matched by its trigrams
  the same way.
* Memories are **ranked** by BM25 over the query's trigrams: every query word
  counts, a trigram found in most memories (``the``, ``use``: no stopword list is
  needed) next to nothing, and a long memory does not win by its length. Ties keep
  the order the memories came in.
* Trigrams make almost everything share something with everything — ``running``
  and "every morning" share ``nin``, ``ing`` and ``ng_`` — so the scores are **cut
  at the low end**: a memory scoring more than :attr:`MemorySearch.cutoff_std`
  standard deviations below the mean of the memories that scored is dropped. High
  scores are always kept. The cut is relative to the query's own scores, since BM25
  scores of two queries cannot be compared.

There is no index to keep. A search folds each memory once and counts only the
query's trigrams in it (``str.count``), which is all BM25 needs. The memories are the
only state, so a file edited by hand is searched as it now is.
"""

from __future__ import annotations

import math
import re
import statistics
import unicodedata
from collections import Counter
from collections.abc import Sequence
from typing import TYPE_CHECKING, ClassVar

from pydantic import BaseModel, ConfigDict, Field

if TYPE_CHECKING:
    from domyn.agents.memory.managers.base import Memory


class MemorySearch(BaseModel):
    """How memories are searched; its fields are the settings, all with defaults.

    Declared on a memory manager as ``search:``, so a configuration can tune it::

        memory_manager:
          type: filesystem
          search: {cutoff_std: 0.5}
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    N: ClassVar[int] = 3
    """Trigrams: short enough that another form of a word still shares most of them."""
    EDGE: ClassVar[str] = "_"
    """Marks where a word starts and ends, in its trigrams and in a folded memory."""
    _WORD: ClassVar[re.Pattern[str]] = re.compile(r"\w+")

    cutoff_std: float = Field(default=1.0, ge=0)
    """Memories scoring more than this many standard deviations below the mean are
    dropped; ``0`` keeps only those at or above the mean."""
    k1: float = Field(default=1.2, ge=0)
    """BM25: how quickly more of the same trigram stops adding to a score."""
    b: float = Field(default=0.75, ge=0, le=1)
    """BM25: how much a memory's length is held against it."""

    # ── text ─────────────────────────────────────────────────────────────────

    @staticmethod
    def fold(text: str) -> str:
        """Lower case, no accents, compatibility forms decomposed; every script kept."""
        if text.isascii():
            return text.lower()
        decomposed = unicodedata.normalize("NFKD", text)
        return "".join(c for c in decomposed if not unicodedata.combining(c)).casefold()

    @classmethod
    def words(cls, text: str) -> list[str]:
        """The words of ``text`` a search compares, folded, in order.

        ``_`` is a word character to a regular expression, but marks a word's edges
        here, so it splits words.
        """
        return cls._WORD.findall(cls.fold(text).replace(cls.EDGE, " "))

    @classmethod
    def trigrams(cls, word: str) -> list[str]:
        """The trigrams of one word, its edges marked: ``_it``, ``ita``, … ``ly_``."""
        marked = f"{cls.EDGE}{word}{cls.EDGE}"
        if len(marked) <= cls.N:
            return [marked]
        return [marked[i : i + cls.N] for i in range(len(marked) - cls.N + 1)]

    # ── search ───────────────────────────────────────────────────────────────

    def search(self, memories: Sequence[Memory], query: str) -> list[Memory]:
        """The memories that match ``query``, best first; ties keep the given order.

        A query with no words leaves ``memories`` as they are.
        """
        asked = self.words(query)
        if not asked:
            return list(memories)
        if not memories:
            return []
        # Words between edge marks, apart: a trigram of a word has no space in it,
        # so none is counted across two words.
        texts = [
            " ".join(f"{self.EDGE}{w}{self.EDGE}" for w in self.words(m.memory)) for m in memories
        ]
        scores = self.scores(texts, Counter(g for w in asked for g in self.trigrams(w)))
        kept = self.above_cutoff(scores)
        return [memories[i] for i in sorted(kept, key=lambda i: (-scores[i], i))]

    def scores(self, texts: list[str], grams: Counter[str]) -> dict[int, float]:
        """BM25 over ``grams`` of every text that holds one of them, by position."""
        total = len(texts)
        average = sum(len(t) for t in texts) / total or 1.0
        scores: dict[int, float] = {}
        for gram, repeats in grams.items():
            holders = {i: n for i, text in enumerate(texts) if (n := text.count(gram))}
            idf = math.log(1 + (total - len(holders) + 0.5) / (len(holders) + 0.5))
            for i, found in holders.items():
                norm = self.k1 * (1 - self.b + self.b * len(texts[i]) / average)
                gain = repeats * idf * found * (self.k1 + 1) / (found + norm)
                scores[i] = scores.get(i, 0.0) + gain
        return scores

    def above_cutoff(self, scores: dict[int, float]) -> list[int]:
        """The positions whose score is not more than ``cutoff_std`` deviations below
        the mean of ``scores``; every one when there are fewer than two.

        Compared with a tolerance: with two scores the lower one lies exactly one
        deviation below the mean, and rounding alone must not decide whether it stays.
        """
        if len(scores) < 2:
            return list(scores)
        values = list(scores.values())
        floor = statistics.fmean(values) - self.cutoff_std * statistics.pstdev(values)
        floor -= 1e-9 * max(values)
        return [i for i, score in scores.items() if score >= floor]
