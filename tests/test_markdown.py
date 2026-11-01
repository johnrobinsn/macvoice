"""What the voice leaves out of a markdown reply."""

import pytest

from macvoice.markdown import plain, without_code_blocks

REPLY = "Here is the fix:\n\n```python\nprint('hi')\n```\n\nThen run it.\n  ```\nindented fence\n  ```\nDone."


@pytest.mark.parametrize("size", [1, 2, 3, 7, 1000])
def test_code_blocks_are_dropped_however_the_text_is_streamed(size):
    tokens = [REPLY[i:i + size] for i in range(0, len(REPLY), size)]
    spoken = "".join(without_code_blocks(tokens))
    assert "print" not in spoken and "indented fence" not in spoken and "```" not in spoken
    assert spoken.startswith("Here is the fix:") and "Then run it." in spoken and spoken.endswith("Done.")


def test_text_streams_through_without_waiting_for_the_line():
    out = list(without_code_blocks(["Hello ", "there, ", "friend"]))
    assert out == ["Hello ", "there, ", "friend"]              # nothing held back mid-line


def test_backticks_that_are_not_fences_pass_through():
    assert "".join(without_code_blocks(["`x` is ", "fine\n`y`"])) == "`x` is fine\n`y`"


@pytest.mark.parametrize("text, said", [
    ("I fixed **two** bugs in `app.py`.", "I fixed two bugs in app.py."),
    ("- Run the [tests](http://example.com).", "Run the tests."),
    ("## Summary", "Summary"),
    ("> quoted", "quoted"),
    ("Use snake_case names.", "Use snake_case names."),       # underscores inside words stay
])
def test_plain(text, said):
    assert plain(text) == said
