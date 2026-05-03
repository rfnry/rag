import logging
import re
from collections.abc import Callable
from typing import Literal

logger = logging.getLogger(__name__)


def _split_with_separator(
    text: str,
    separator: str,
    keep_separator: bool | Literal["start", "end"],
) -> list[str]:
    """Split text on a separator, optionally keeping the separator attached to chunks."""
    if not separator:
        return list(text)

    escaped = re.escape(separator)

    if keep_separator:
        parts = re.split(f"({escaped})", text)
        splits: list[str] = []
        if keep_separator == "end":
            for i in range(0, len(parts) - 1, 2):
                splits.append(parts[i] + parts[i + 1])
            if len(parts) % 2 == 1:
                splits.append(parts[-1])
        else:
            splits.append(parts[0])
            for i in range(1, len(parts) - 1, 2):
                splits.append(parts[i] + parts[i + 1])
            if len(parts) % 2 == 0:
                splits.append(parts[-1])
    else:
        splits = re.split(escaped, text)

    return [s for s in splits if s]


class RecursiveTextSplitter:
    """Recursive text splitter that tries separators in priority order.

    Splits text by trying each separator in order, keeping chunks under
    chunk_size with configurable overlap. Separators default to
    ``["\\n\\n", "\\n", ". ", " ", ""]`` — paragraph, line, sentence,
    word, then character boundaries.
    """

    def __init__(
        self,
        chunk_size: int = 500,
        chunk_overlap: int = 50,
        separators: list[str] | None = None,
        keep_separator: bool | Literal["start", "end"] = False,
        length_function: Callable[[str], int] = len,
        strip_whitespace: bool = True,
    ) -> None:
        self._chunk_size = chunk_size
        self._chunk_overlap = chunk_overlap
        self._separators = separators or ["\n\n", "\n", ". ", " ", ""]
        self._keep_separator = keep_separator
        self._length_function = length_function
        self._strip_whitespace = strip_whitespace

    def split_text(self, text: str) -> list[str]:
        return [chunk for chunk, _ in self.split_text_with_flags(text)]

    def split_text_with_flags(self, text: str) -> list[tuple[str, bool]]:
        """Split text and return (chunk_text, was_hard_split) tuples.

        ``was_hard_split`` is True when no natural word/sentence boundary was
        available and character-level splitting was required. Two code paths
        set the flag:

        * **Char-level merge** — the separator ladder exhausted to ``""`` and
          all chunks produced from that level are flagged, even those that fit
          within ``chunk_size`` after merging.
        * **Hard-slice loop** — the ladder found no separator at all (possible
          with custom ``separators`` lists that omit ``""``) and a remaining
          piece exceeded ``chunk_size``; it is force-sliced at ``chunk_size``
          character boundaries.
        """
        return self._split_flagged(text, self._separators)

    def _join(self, pieces: list[str], separator: str) -> str | None:
        """Join pieces with separator and optionally strip whitespaces."""
        joiner = "" if self._keep_separator else separator
        text = joiner.join(pieces)
        if self._strip_whitespace:
            text = text.strip()
        return text or None

    def _merge_splits(self, splits: list[str], separator: str) -> list[str]:
        """Greedily pack small splits into chunks up to chunk_size with overlap.

        Overlap is implemented by popping from the front of the buffer
        until retained content <= chunk_overlap and the new piece fits.
        """
        joiner = "" if self._keep_separator else separator
        joiner_len = self._length_function(joiner)

        chunks: list[str] = []
        current: list[str] = []
        total = 0

        for piece in splits:
            piece_len = self._length_function(piece)
            candidate = total + piece_len + (joiner_len if current else 0)

            if candidate > self._chunk_size and current:
                if total > self._chunk_size:
                    logger.warning(
                        "Created a chunk of size %d, which is longer than the specified %d",
                        total,
                        self._chunk_size,
                    )
                joined = self._join(current, separator)
                if joined is not None:
                    chunks.append(joined)

                while total > self._chunk_overlap or (
                    total + piece_len + (joiner_len if current else 0) > self._chunk_size and total > 0
                ):
                    removed_len = self._length_function(current[0]) + (joiner_len if len(current) > 1 else 0)
                    total -= removed_len
                    current = current[1:]

            current.append(piece)
            total += piece_len + (joiner_len if len(current) > 1 else 0)

        joined = self._join(current, separator)
        if joined is not None:
            chunks.append(joined)

        return chunks

    def _split_flagged(self, text: str, separators: list[str]) -> list[tuple[str, bool]]:
        """Recursively split text, returning (chunk_text, was_hard_split) pairs.

        When the separator ladder is exhausted and a piece still exceeds
        ``chunk_size``, it is force-sliced at ``chunk_size`` boundaries and
        each resulting slice is flagged with ``was_hard_split=True``.
        """
        if not text:
            return []

        if self._length_function(text) <= self._chunk_size:
            stripped = text.strip() if self._strip_whitespace else text
            return [(stripped, False)] if stripped else []

        separator = separators[-1]
        remaining_separators: list[str] = []
        for i, sep in enumerate(separators):
            if not sep:
                separator = sep
                break
            if sep in text:
                separator = sep
                remaining_separators = separators[i + 1 :]
                break

        # Splitting at the empty-string (character) level means we've exhausted
        # all natural separators — any chunks produced at this level are hard-splits.
        is_char_level = separator == ""

        splits = _split_with_separator(text, separator, self._keep_separator)

        flagged: list[tuple[str, bool]] = []
        good_splits: list[str] = []

        for piece in splits:
            if self._length_function(piece) < self._chunk_size:
                good_splits.append(piece)
            else:
                if good_splits:
                    hard = is_char_level
                    flagged.extend((c, hard) for c in self._merge_splits(good_splits, separator))
                    good_splits = []

                if remaining_separators:
                    flagged.extend(self._split_flagged(piece, remaining_separators))
                else:
                    # No separators left and piece still exceeds chunk_size:
                    # hard-split at chunk_size boundaries.
                    if self._length_function(piece) > self._chunk_size:
                        step = max(1, self._chunk_size - self._chunk_overlap)
                        start = 0
                        while start < len(piece):
                            hard_slice = piece[start : start + self._chunk_size]
                            if self._strip_whitespace:
                                hard_slice = hard_slice.strip()
                            if hard_slice:
                                flagged.append((hard_slice, True))
                            start += step
                    else:
                        flagged.append((piece, is_char_level))

        if good_splits:
            hard = is_char_level
            flagged.extend((c, hard) for c in self._merge_splits(good_splits, separator))

        return flagged
