"""Startup checks that run before module imports."""

import sys


def check_baml(sdk: str, baml_client_module: str) -> None:
    """Validate baml-py version matches the generated client before importing modules.

    Args:
        sdk: SDK name for error messages (e.g. "retrieval", "reasoning").
        baml_client_module: Dotted module path to the BAML client package
            (e.g. "rfnry_rag.retrieval.baml.baml_client").
    """
    try:
        import importlib
        from importlib.metadata import version as pkg_version

        client_mod = importlib.import_module(baml_client_module)
        generator_version: str = client_mod.__version__  # type: ignore[attr-defined]
        installed = pkg_version("baml-py")
    except ImportError as exc:
        print(
            f"rfnry-rag.{sdk}: baml dependency error:\n  {exc}\n\nrun: uv sync --all-extras",
            file=sys.stderr,
        )
        sys.exit(1)
    except Exception:
        return

    gen_minor = ".".join(generator_version.split(".")[:2])
    inst_minor = ".".join(installed.split(".")[:2])

    if gen_minor != inst_minor:
        print(
            f"rfnry-rag.{sdk}: baml version mismatch:\n"
            f"  generator (generators.baml): {generator_version}\n"
            f"  installed (baml-py):         {installed}\n\n"
            f"run: uv add baml-py=={generator_version}\n"
            f"  or: uv run poe baml:generate  (to regenerate client for {installed})",
            file=sys.stderr,
        )
        sys.exit(1)
