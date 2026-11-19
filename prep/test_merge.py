"""The merge rule, pinned: one conversation read twice overlaps, two conversations do not.

`MIN_OVERLAP = 0.5` was a number with nothing behind it. It matters more here than in the donor tool,
because the join cannot use names — they are the thing the mask removes — so the only evidence that two
payloads are the same conversation is the messages they share. Half the keys is the line between "a
second read of this chat" (which shares nearly everything) and "somebody else, same week" (which shares a
`hej` or two).

Run: `uv run python -B -m pytest prep/ -q` (or `python -B prep/test_merge.py`).
"""
from __future__ import annotations

import hashlib

try:  # as a module (`python -m prep.test_merge`) or as a file (`python prep/test_merge.py`)
    from .merge import MIN_OVERLAP, identity_signature, message_key, merge_payloads, overlaps
except ImportError:  # pragma: no cover
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from prep.merge import MIN_OVERLAP, identity_signature, message_key, merge_payloads, overlaps


def msg(ts: int, text: str) -> dict:
    return {"ts_ms": ts, "text": text, "sender": "SUBJECT"}


def payload(messages: list[dict], platform: str = "fb", partner: str = "OTHER-004") -> dict:
    """A bundle-shaped conversation. `partner` matters: two chats with the same two people *are* one
    conversation, so a test about two different chats has to give them different people."""
    return {"platform": platform, "participants": ["SUBJECT", partner], "messages": messages}


def test_a_second_read_of_one_conversation_merges() -> None:
    # Two exports of the same chat: the dump has everything, the cutover has the newer half, and both
    # hold a run of messages in common. That overlap is the whole signal.
    first = payload([msg(t, f"line {t}") for t in range(1, 21)])
    second = payload([msg(t, f"line {t}") for t in range(12, 26)])
    merged = merge_payloads([("fb-e2ee-dump", first), ("fb-e2ee_cutover", second)])

    assert len(merged) == 1, "one conversation, read twice"
    assert merged[0]["sources"] == ["fb-e2ee-dump", "fb-e2ee_cutover"]
    assert [m["ts_ms"] for m in merged[0]["messages"]] == list(range(1, 26)), "sorted, and no duplicate"
    assert merged[0]["n_messages"] == 25


def test_reads_with_no_shared_messages_merge_on_the_fingerprint() -> None:
    # A device switch splits one conversation in time: the dump has the first half, the cutover the
    # second, and they share nothing. Content overlap scores 0 and refuses; the identity fingerprint
    # is what makes this the same conversation. This is the case that made a name-based merge necessary
    # in the first place, and the reason the fingerprint travels in the bundle.
    sig = identity_signature(["Elias Test", "Airidas Brikas"])
    assert sig == identity_signature([" airidas brikas ", "ELIAS TEST"]), "order and case do not matter"
    old = payload([msg(t, f"old {t}") for t in range(1, 11)])
    new = payload([msg(t, f"new {t}") for t in range(100, 110)])
    old["identity_sig"] = new["identity_sig"] = sig
    merged = merge_payloads([("fb-e2ee-dump", old), ("fb-e2ee_cutover", new)])
    assert len(merged) == 1, "one conversation, read twice, disjoint in time"
    assert merged[0]["n_messages"] == 20


def test_two_conversations_from_one_week_stay_apart() -> None:
    # Same person, same week, different chats. Sharing a greeting must not be enough.
    one = payload([msg(t, f"alpha {t}") for t in range(1, 11)], partner="OTHER-001")
    two = payload([msg(t, f"beta {t}") for t in range(1, 11)], partner="OTHER-002")
    merged = merge_payloads([("fb-a", one), ("fb-b", two)])
    assert len(merged) == 2, "different messages are different conversations"
    assert sorted(p["sources"][0] for p in merged) == ["fb-a", "fb-b"]


def test_platform_is_part_of_identity() -> None:
    # Identical text and timestamps on two channels are two conversations: a shared link can be sent to
    # somebody on Facebook and to somebody else on Instagram, and labels are per channel.
    same = [msg(t, f"line {t}") for t in range(1, 11)]
    merged = merge_payloads([("fb-x", payload(same, "fb")), ("ig-x", payload(same, "ig"))])
    assert len(merged) == 2


def test_the_threshold_is_half_and_that_is_what_it_means() -> None:
    # Pinned deliberately: a change to MIN_OVERLAP is a change to how much evidence sameness needs, and
    # should be an edit to this line rather than a silent shift in behaviour.
    assert MIN_OVERLAP == 0.5

    a = frozenset(message_key(m) for m in [msg(t, f"a {t}") for t in range(1, 11)])
    # Half of the smaller one is inside the larger: exactly at the line, and it merges.
    b = frozenset(list(a)[:5] + [message_key(msg(t, f"b {t}")) for t in range(1, 6)])
    assert overlaps(a, b) == 0.5
    # One shared line: a greeting, not a conversation.
    c = frozenset(list(a)[:1] + [message_key(msg(t, f"c {t}")) for t in range(1, 18)])
    assert overlaps(a, c) < MIN_OVERLAP
    # Four shared messages is under MIN_SHARED, however small the payload: 4 of 4 is "identical", and
    # two four-message threads are not evidence of anything but a repeated "hej".
    tiny = frozenset(list(a)[:4])
    assert overlaps(tiny, tiny) == 0.0


def test_an_empty_payload_never_merges() -> None:
    # A thread with no messages shares nothing with anything, including another empty one. Two empty
    # payloads are not evidence of the same conversation; they are evidence of nothing.
    assert overlaps(frozenset(), frozenset()) == 0.0
    merged = merge_payloads([("a", payload([], partner="OTHER-001")), ("b", payload([], partner="OTHER-002"))])
    assert len(merged) == 2


def test_a_message_key_ignores_everything_but_time_and_words() -> None:
    # Timestamps and text are what survive masking. Sender is a label that differs between donations,
    # and reactions are not part of identity: the same message with a heart added is the same message.
    a = {"ts_ms": 1000, "text": "hej", "sender": "OTHER-004", "reactions": []}
    b = {"ts_ms": 1000, "text": "hej", "sender": "SUBJECT", "reactions": [{"emoji": "❤", "actor": "SUBJECT"}]}
    assert message_key(a) == message_key(b)
    assert message_key(a) != message_key({"ts_ms": 1001, "text": "hej"})
    assert hashlib.sha1(b"hej").hexdigest()[:16] == message_key(a)[1]


if __name__ == "__main__":
    failures = 0
    for name, case in sorted(globals().items()):
        if name.startswith("test_") and callable(case):
            try:
                case()
                print(f"  ok   {name}")
            except AssertionError as problem:
                failures += 1
                print(f"  FAIL {name}: {problem}")
    print(f"\n{'all good' if not failures else f'{failures} failing'}")
    raise SystemExit(1 if failures else 0)
