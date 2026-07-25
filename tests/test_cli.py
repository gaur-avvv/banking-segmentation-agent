import json

from banking_agent.cli import _parser, _trace_plan


def test_chat_parser_preserves_data_directory_and_trace_setting():
    args = _parser().parse_args(["chat", "--data-dir", "demo", "--trace", "off"])
    assert args.command == "chat"
    assert args.data_path == "demo"
    assert args.trace == "off"


def test_trace_plan_extracts_intent_from_event():
    events = [{"step": "query_planning", "detail": json.dumps({"intent": "compare_segments"})}]
    assert _trace_plan(events) == "compare_segments"
