"""Startup check: baml-py runtime version matches the generated client."""

import sys


def check_baml() -> None:
    """Validate baml-py version matches the generated client before module imports."""
    try:
        import importlib
        from importlib.metadata import version as pkg_version

        client_mod = importlib.import_module("rfnry_knowledge.baml.baml_client")
        generator_version: str = client_mod.__version__  # type: ignore[attr-defined]
        installed = pkg_version("baml-py")
    except ImportError as exc:
        print(
            f"rfnry-knowledge: baml dependency error:\n  {exc}\n\nrun: uv sync --all-extras",
            file=sys.stderr,
        )
        sys.exit(1)
    except Exception:
        return

    gen_minor = ".".join(generator_version.split(".")[:2])
    inst_minor = ".".join(installed.split(".")[:2])

    if gen_minor != inst_minor:
        print(
            f"rfnry-knowledge: baml version mismatch:\n"
            f"  generator (generators.baml): {generator_version}\n"
            f"  installed (baml-py):         {installed}\n\n"
            f"run: uv add baml-py=={generator_version}\n"
            f"  or: uv run poe baml:generate  (to regenerate client for {installed})",
            file=sys.stderr,
        )
        sys.exit(1)
