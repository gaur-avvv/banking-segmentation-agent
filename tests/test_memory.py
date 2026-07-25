from banking_agent.memory import MemoryAwareRecommendation, MemoryEntry, SQLiteMemoryStore


def test_memory_is_customer_scoped_deterministic_and_forgettable(tmp_path):
    store = SQLiteMemoryStore(tmp_path / "memory.db")
    first = MemoryEntry("user-a", "query", {"topic": "credit card"}, {"contact_method": "chat"}, {}, True, products_mentioned=["card"])
    second = MemoryEntry("user-b", "query", {"topic": "mortgage"}, {}, {}, True)
    store.save(first)
    store.save(second)
    assert [m.memory_id for m in store.search("user-a", "credit card")] == [first.memory_id]
    assert store.forget_user("user-a") == 1
    assert store.search("user-a", "credit card") == []


def test_memory_requires_consent_and_does_not_override_ineligibility(tmp_path):
    store = SQLiteMemoryStore(tmp_path / "memory.db")
    try:
        store.save(MemoryEntry("user-a", "query", {}, {}, {}, False))
        assert False, "Expected missing consent to be rejected"
    except PermissionError:
        pass
    store.save(MemoryEntry("user-a", "query", {"topic": "card"}, {}, {}, True, products_mentioned=["card", "card"]))
    profile = store.build_profile("user-a")
    adjusted = MemoryAwareRecommendation.adjust([{"product": "card", "eligible": False}], profile)
    assert adjusted[0]["memory_rank_adjustment"] == 0
