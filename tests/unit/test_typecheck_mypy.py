import tempfile
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit


@pytest.mark.skipif(
    not bool(__import__("importlib").util.find_spec("mypy")),
    reason="mypy not installed",
)
def test_mypy_catches_misuse_of_auto_context_manager() -> None:
    """
    Ensure type checker flags using the auto-detecting context manager with
    a plain `with` or `async with` due to its Union return type.
    """
    from mypy import api as mypy_api  # type: ignore

    code = (
        "from pharia_telemetry.sem_conv.gen_ai import GenAI, create_genai_span, "
        "create_genai_span_sync, create_genai_span_async\n\n"
        "def good_sync() -> None:\n"
        "    with create_genai_span_sync(GenAI.Values.OperationName.CHAT):\n"
        "        pass\n\n"
        "async def good_async() -> None:\n"
        "    async with create_genai_span_async(GenAI.Values.OperationName.CHAT):\n"
        "        pass\n\n"
        "def bad_sync() -> None:\n"
        "    with create_genai_span(GenAI.Values.OperationName.CHAT):\n"
        "        pass\n\n"
        "async def bad_async() -> None:\n"
        "    async with create_genai_span(GenAI.Values.OperationName.CHAT):\n"
        "        pass\n"
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpfile = Path(tmpdir) / "sample.py"
        tmpfile.write_text(code)

        # Run mypy against the temporary file with project config
        stdout, stderr, exit_status = mypy_api.run([str(tmpfile)])

        # We expect mypy to report errors for misuse
        assert exit_status != 0, (
            "mypy did not flag misuse of auto-detecting context manager.\n"
            f"stdout:\n{stdout}\nstderr:\n{stderr}"
        )

        # Robust check: mypy should complain about missing __exit__/__aexit__ on a union
        assert (
            ("__exit__" in stdout)
            or ("__aexit__" in stdout)
            or ("union-attr" in stdout)
        )
