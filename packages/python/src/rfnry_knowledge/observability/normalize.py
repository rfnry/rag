import re
import string


def normalize_answer(s: str) -> str:
    """Normalize text for evaluation comparison.

    Standard SQuAD-style normalization: lowercase, strip punctuation,
    remove articles (a/an/the), collapse whitespaces.
    """

    def remove_articles(text: str) -> str:
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def remove_punctuation(text: str) -> str:
        return "".join(ch for ch in text if ch not in string.punctuation)

    def collapse_whitespace(text: str) -> str:
        return " ".join(text.split())

    return collapse_whitespace(remove_articles(remove_punctuation(s.lower())))
