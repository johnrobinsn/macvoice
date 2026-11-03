"""A scripted ACP agent for tests: speaks JSON-RPC on stdio like
claude-agent-acp. Prompts containing "tool" ask permission first and report
the outcome; "slow" streams forever until session/cancel; "freeze" sends nothing until
session/cancel; "edit" reports tool calls; "die" exits. It
advertises three modes unless started with "nomodes", and model and effort
settings (configOptions, as claude-agent-acp does) unless "noconfig"."""

import json
import sys

LOG = sys.argv[1] if len(sys.argv) > 1 else None
MODES = None if "nomodes" in sys.argv else {
    "currentModeId": "default",
    "availableModes": [{"id": "default", "name": "Default"},
                       {"id": "acceptEdits", "name": "Accept Edits", "description": "edits go through"},
                       {"id": "plan", "name": "Plan Mode"}]}

CONFIG = None if "noconfig" in sys.argv else [
    {"id": "model", "name": "Model", "category": "model", "type": "select", "currentValue": "default",
     "options": [{"value": "default", "name": "Default (recommended)"},
                 {"value": "opus", "name": "Opus 5.5"}, {"value": "sonnet", "name": "Sonnet 5.5"}]},
    {"id": "effort", "name": "Effort", "category": "thought_level", "type": "select",
     "currentValue": "default",
     "options": [{"value": v, "name": v.title()} for v in ("default", "low", "medium", "high")]}]


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
        send({"id": mid, "result": {"sessionId": "s1", **({"modes": MODES} if MODES else {}),
                                    **({"configOptions": CONFIG} if CONFIG else {})}})
    elif method == "session/set_mode":
        send({"id": mid, "result": {}})
    elif method == "session/set_config_option":
        option = next((o for o in CONFIG or [] if o["id"] == params["configId"]), None)
        if option is None or params["value"] not in [c["value"] for c in option["options"]]:
            send({"id": mid, "error": {"code": -32603, "message": "Internal error"}})
        else:
            option["currentValue"] = params["value"]
            send({"id": mid, "result": {"configOptions": CONFIG}})
    elif method == "session/prompt":
        text = params["prompt"][0]["text"]
        sid = params["sessionId"]
        if "die" in text:
            sys.exit(0)
        send({"method": "session/update", "params": {"sessionId": sid, "update": {
            "sessionUpdate": "agent_thought_chunk", "content": {"type": "text", "text": "hmm"}}}})
        if "tool" in text:
            send({"id": 900, "method": "session/request_permission", "params": {
                "sessionId": sid, "toolCall": {"title": "`npm install`", "kind": "execute",
                                               "rawInput": {"command": "npm install"}},
                "options": [{"optionId": "yes", "kind": "allow_once", "name": "Allow"},
                            {"optionId": "no", "kind": "reject_once", "name": "Reject"}]}})
            answer = read()
            log(answer)
            chunk(sid, f"Permission {answer['result']['outcome'].get('optionId')}. ")
        if "edit" in text:
            for update in ({"sessionUpdate": "tool_call", "toolCallId": "t1", "title": "Edit `index.html`",
                            "kind": "edit", "status": "pending"},
                           {"sessionUpdate": "tool_call_update", "toolCallId": "t1", "status": "in_progress"},
                           {"sessionUpdate": "tool_call_update", "toolCallId": "t1", "status": "completed"},
                           {"sessionUpdate": "tool_call", "toolCallId": "t2", "title": "npm test",
                            "kind": "execute"},
                           {"sessionUpdate": "tool_call_update", "toolCallId": "t2", "status": "failed"}):
                send({"method": "session/update", "params": {"sessionId": sid, "update": update}})
        if "freeze" in text:                      # a long silent tool: nothing until session/cancel
            cancel = read()
            log(cancel)
            send({"id": mid, "result": {"stopReason": "cancelled"}})
            continue
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
