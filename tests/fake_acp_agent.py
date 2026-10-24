"""A scripted ACP agent for tests: speaks JSON-RPC on stdio like
claude-agent-acp. Prompts containing "tool" ask permission first and report
the outcome; "slow" streams forever until session/cancel; "die" exits."""

import json
import sys

LOG = sys.argv[1] if len(sys.argv) > 1 else None


def send(msg):
    sys.stdout.write(json.dumps({"jsonrpc": "2.0", **msg}) + "\n")
    sys.stdout.flush()


def log(entry):
    if LOG:
        with open(LOG, "a") as f:
            f.write(json.dumps(entry) + "\n")


def chunk(sid, text):
    send({"method": "session/update", "params": {"sessionId": sid, "update": {
        "sessionUpdate": "agent_message_chunk", "content": {"type": "text", "text": text}}}})


def read():
    line = sys.stdin.readline()
    return json.loads(line) if line else None


while (msg := read()) is not None:
    log(msg)
    method, mid, params = msg.get("method"), msg.get("id"), msg.get("params", {})
    if method == "initialize":
        send({"id": mid, "result": {"protocolVersion": 1, "agentInfo": {"name": "fake"}}})
    elif method == "session/new":
        send({"id": mid, "result": {"sessionId": "s1"}})
    elif method == "session/prompt":
        text = params["prompt"][0]["text"]
        sid = params["sessionId"]
        if "die" in text:
            sys.exit(0)
        send({"method": "session/update", "params": {"sessionId": sid, "update": {
            "sessionUpdate": "agent_thought_chunk", "content": {"type": "text", "text": "hmm"}}}})
        if "tool" in text:
            send({"id": 900, "method": "session/request_permission", "params": {
                "sessionId": sid, "toolCall": {"title": "Bash: ls"},
                "options": [{"optionId": "yes", "kind": "allow_once", "name": "Allow"},
                            {"optionId": "no", "kind": "reject_once", "name": "Reject"}]}})
            answer = read()
            log(answer)
            chunk(sid, f"Permission {answer['result']['outcome'].get('optionId')}. ")
        if "slow" in text:
            n = 0
            while True:
                chunk(sid, f"word{n} ")
                n += 1
                if n % 50 == 0:
                    import select
                    if select.select([sys.stdin], [], [], 0.01)[0]:
                        cancel = read()
                        log(cancel)
                        send({"id": mid, "result": {"stopReason": "cancelled"}})
                        break
            continue
        for part in ("Paris is ", "the capital. ", "Anything else?"):
            chunk(sid, part)
        send({"id": mid, "result": {"stopReason": "end_turn"}})
    elif mid is not None:
        send({"id": mid, "error": {"code": -32601, "message": "nope"}})
