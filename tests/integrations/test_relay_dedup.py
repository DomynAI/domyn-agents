from domyn.agents.integrations.utils import SeenWindow, resume_headers

# ---------------------------------------------------------------------------
# resume_headers — persistent last-event-id cursor
# ---------------------------------------------------------------------------


def test_resume_headers_omits_cursor_when_none():
    assert resume_headers({"channel-id": "c"}, None) == {"channel-id": "c"}


def test_resume_headers_includes_cursor_when_set():
    assert resume_headers({"channel-id": "c"}, "e1") == {
        "channel-id": "c",
        "last-event-id": "e1",
    }


def test_resume_headers_does_not_mutate_base():
    base = {"channel-id": "c"}
    resume_headers(base, "e1")
    assert "last-event-id" not in base


# ---------------------------------------------------------------------------
# SeenWindow — unified sent+seen dedup, bounded by count + TTL
# ---------------------------------------------------------------------------


class _Clock:
    """Manually-advanced monotonic clock for deterministic TTL tests."""

    def __init__(self) -> None:
        self.t = 0.0

    def __call__(self) -> float:
        return self.t


def test_seen_window_first_new_then_duplicate():
    w = SeenWindow()
    assert w.is_new("e1") is True
    assert w.is_new("e1") is False  # replayed duplicate


def test_seen_window_add_then_is_new_drops_own_echo():
    w = SeenWindow()
    w.add("e1")  # we sent it
    assert w.is_new("e1") is False  # relay echoes it back → dropped


def test_seen_window_falsy_id_always_new_and_add_is_noop():
    w = SeenWindow()
    assert w.is_new(None) is True
    assert w.is_new("") is True
    w.add(None)
    w.add("")
    # nothing recorded, so a real id is still new
    assert w.is_new("e1") is True


def test_seen_window_evicts_oldest_beyond_max_count():
    w = SeenWindow(max_count=3)
    for eid in ("e1", "e2", "e3"):
        assert w.is_new(eid) is True
    # e4 pushes e1 (oldest) out of the window.
    assert w.is_new("e4") is True
    assert w.is_new("e1") is True  # forgotten → treated as new again
    # e1's reinsertion pushes e2 (now the oldest) out of the window too.
    assert w.is_new("e2") is True


def test_seen_window_expires_entries_after_ttl():
    clock = _Clock()
    w = SeenWindow(ttl_seconds=100.0, now=clock)
    assert w.is_new("e1") is True
    clock.t = 50.0
    assert w.is_new("e1") is False  # still fresh
    clock.t = 101.0
    assert w.is_new("e1") is True  # expired → new again


def test_seen_window_add_does_not_extend_deadline_on_reuse():
    """The cancel path re-sends an already-seen id; re-add must not refresh its TTL."""
    clock = _Clock()
    w = SeenWindow(ttl_seconds=100.0, now=clock)
    w.add("e1")  # deadline = 100
    clock.t = 60.0
    w.add("e1")  # setdefault → deadline stays 100, not 160
    clock.t = 101.0
    assert w.is_new("e1") is True  # expired at 100 as originally scheduled
