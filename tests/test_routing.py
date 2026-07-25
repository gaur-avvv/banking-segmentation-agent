from banking_agent.routing import route_query


def test_router_is_deterministic_and_prefers_conversion_intent():
    query = "Which regular customers can be converted to priority?"
    assert route_query(query) == route_query(query)
    assert route_query(query)["intent"] == "recommend_conversion"


def test_router_marks_unknown_request_for_review():
    assert route_query("Make this better")["needs_human_input"] is True
