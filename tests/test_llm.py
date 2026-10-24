import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

from macvoice.llm import SYSTEM_PROMPT, Echo, OpenAICompatible, parse_sse, sentences


def sse(*deltas, done=True):
    lines = [f"data: {json.dumps({'choices': [{'delta': {'content': d}}]})}\n".encode() for d in deltas]
    return [b": keep-alive\n", b"\n", *lines, *([b"data: [DONE]\n"] if done else [])]


def test_parse_sse_yields_content_deltas():
    lines = sse("Hel", "lo") + [b'data: {"choices": [{"delta": {"content": "late"}}]}\n']
    assert list(parse_sse(lines)) == ["Hel", "lo"]          # stops at [DONE]


def test_parse_sse_skips_role_chunks_and_bad_json():
    lines = [b'data: {"choices": [{"delta": {"role": "assistant"}}]}\n', b"data: {nope\n",
             *sse("ok")]
    assert list(parse_sse(lines)) == ["ok"]


@pytest.mark.parametrize("tokens,expected", [
    (["Hello there. How are you today?"], ["Hello there.", "How are you today?"]),
    (["Hi. ", "I can help with that. ", "Sure!"], ["Hi. I can help with that.", "Sure!"]),
    (["Dr. Smith will see you now. Bye now."], ["Dr. Smith will see you now.", "Bye now."]),
    (["No punctuation at all"], ["No punctuation at all"]),
    (["It costs 3.50 dollars. Okay?"], ["It costs 3.50 dollars.", "Okay?"]),
    (['He said "stop." Then he left.'], ['He said "stop."', "Then he left."]),
    ([], []),
])
def test_sentences(tokens, expected):
    assert list(sentences(tokens)) == expected


def test_sentences_stream_token_by_token():
    text = "The capital of France is Paris. It has about two million people."
    assert list(sentences(list(text))) == ["The capital of France is Paris.",
                                           "It has about two million people."]


def test_sentences_yield_before_stream_ends():
    def tokens():
        yield "First sentence is done. "
        raise AssertionError("consumed past the first sentence")

    assert next(sentences(tokens())) == "First sentence is done."


def test_echo():
    assert list(Echo().reply([{"role": "user", "content": "hi"}])) == ["You said: hi"]


@pytest.fixture
def server():
    seen = {}

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            seen["path"] = self.path
            seen["auth"] = self.headers.get("Authorization")
            seen["body"] = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.end_headers()
            for line in sse("Paris is ", "the capital. ", "Anything else?"):
                self.wfile.write(line)
                self.wfile.flush()

        def log_message(self, *a):
            pass

    httpd = HTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{httpd.server_address[1]}/v1", seen
    httpd.shutdown()


def test_openai_compatible_streams_and_sends_system_prompt(server):
    url, seen = server
    llm = OpenAICompatible(base_url=url + "/", model="m", api_key="k")
    history = [{"role": "user", "content": "capital of France?"}]
    assert list(sentences(llm.reply(history))) == ["Paris is the capital.", "Anything else?"]
    assert seen["path"] == "/v1/chat/completions"
    assert seen["auth"] == "Bearer k"
    assert seen["body"]["stream"] is True and seen["body"]["model"] == "m"
    assert seen["body"]["messages"] == [{"role": "system", "content": SYSTEM_PROMPT}, *history]


def test_openai_compatible_without_key_sends_no_auth(server):
    url, seen = server
    list(OpenAICompatible(base_url=url).reply([{"role": "user", "content": "x"}]))
    assert seen["auth"] is None
