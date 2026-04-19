import hashlib
from pathlib import Path


def file_hash(file_path: str | Path) -> str:
    sha256 = hashlib.sha256()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            sha256.update(chunk)
    return sha256.hexdigest()


def embedding_model_fingerprint(provider: str, model: str) -> str:
    return f"{provider}:{model}"
