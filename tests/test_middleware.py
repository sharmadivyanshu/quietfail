"""QuietfailMiddleware against a real `create_agent`, driven by a fake model.

No API key and no network: a scripted chat model plays back a tool call and
then a final answer, which is enough to exercise every hook.
"""

import pytest

pytest.importorskip("langchain", reason="middleware adapter needs langchain>=1.0")

from langchain.agents import create_agent  # noqa: E402
from langchain_core.language_models import BaseChatModel  # noqa: E402
from langchain_core.messages import AIMessage  # noqa: E402
from langchain_core.outputs import ChatGeneration, ChatResult  # noqa: E402
from langchain_core.tools import tool  # noqa: E402

from quietfail import Budget, Store, build_profile  # noqa: E402
from quietfail.middleware import QuietfailMiddleware  # noqa: E402

CALLS = {"lookup": 0}


@tool
def lookup(query: str) -> str:
    """Look something up."""
    CALLS["lookup"] += 1
    return "" if CALLS["lookup"] > 90 else f"result for {query}"


@tool
def empty_lookup(query: str) -> str:
    """Always comes back with nothing — the silent-failure shape."""
    return ""


class ScriptedChatModel(BaseChatModel):
    """Minimal tool-calling chat model: one tool call, then a final answer.

    langchain's bundled fakes do not implement tool calling, and create_agent
    needs it, so this is the smallest thing that does.
    """

    tool_name: str = "lookup"

    @property
    def _llm_type(self) -> str:
        return "scripted"

    def bind_tools(self, tools, **kwargs):
        return self

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        already_called = any(getattr(m, "type", None) == "tool" for m in messages)
        if already_called:
            message = AIMessage(
                content="Here is the answer about acme office supplies.",
                usage_metadata={"input_tokens": 10, "output_tokens": 8, "total_tokens": 18},
            )
        else:
            message = AIMessage(
                content="",
                tool_calls=[{"name": self.tool_name, "args": {"query": "acme"}, "id": "c1"}],
                usage_metadata={"input_tokens": 10, "output_tokens": 5, "total_tokens": 15},
            )
        return ChatResult(generations=[ChatGeneration(message=message)])


def scripted_model(tool_name: str = "lookup"):
    return ScriptedChatModel(tool_name=tool_name)


def run_agent(store, *, tool_name="lookup", tools=None, **kw):
    middleware = QuietfailMiddleware(
        store,
        agent="fake_agent",
        output_text=lambda s: s["messages"][-1].content if s.get("messages") else "",
        on_alert=lambda a: None,
        **kw,
    )
    agent = create_agent(
        model=scripted_model(tool_name),
        tools=tools or [lookup, empty_lookup],
        middleware=[middleware],
    )
    agent.invoke({"messages": [("user", "look up acme")]})
    return middleware


def test_run_is_recorded(tmp_path):
    store = Store(tmp_path / "m.sqlite")
    run_agent(store)

    runs = store.runs("fake_agent")
    assert len(runs) == 1
    assert runs[0]["status"] == "ok"
    assert runs[0]["duration_ms"] is not None


def test_trajectory_records_model_and_tool_steps(tmp_path):
    store = Store(tmp_path / "m.sqlite")
    run_agent(store)

    path = store.runs("fake_agent")[0]["node_path"]
    assert path[0] == "model"
    assert "tool:lookup" in path


def test_tool_events_are_captured(tmp_path):
    store = Store(tmp_path / "m.sqlite")
    run_agent(store)

    events = store.runs("fake_agent")[0]["tool_events"]
    assert [e["tool"] for e in events] == ["lookup"]
    assert events[0]["errored"] == 0
    assert events[0]["latency_ms"] >= 0


def test_empty_tool_result_is_marked_empty(tmp_path):
    """The whole point of the tool signal: [] is not an error, but it is a
    failure."""
    store = Store(tmp_path / "m.sqlite")
    run_agent(store, tool_name="empty_lookup")

    events = store.runs("fake_agent")[0]["tool_events"]
    assert events[0]["tool"] == "empty_lookup"
    assert events[0]["empty"] == 1
    assert events[0]["errored"] == 0


def test_output_text_is_captured_for_content_drift(tmp_path):
    store = Store(tmp_path / "m.sqlite")
    run_agent(store)
    assert "acme" in store.runs("fake_agent")[0]["output_text"]


def test_default_outcome_reflects_whether_it_answered(tmp_path):
    store = Store(tmp_path / "m.sqlite")
    run_agent(store)
    assert store.runs("fake_agent")[0]["outcome"] == "answered"


def test_budget_is_charged(tmp_path):
    store = Store(tmp_path / "m.sqlite")
    run_agent(store, budget_factory=lambda: Budget(max_steps=10))
    # Two model calls and one tool call in the script.
    assert store.runs("fake_agent")[0]["tool_calls"] == 1


def test_budget_breach_stops_the_run(tmp_path):
    from quietfail.budget import BudgetExceeded

    store = Store(tmp_path / "m.sqlite")
    with pytest.raises(BudgetExceeded):
        run_agent(store, budget_factory=lambda: Budget(max_steps=1))


def test_alerts_fire_against_a_baseline(tmp_path):
    """Record healthy runs, build a baseline, then break the tool and check
    the same signals used by the StateGraph path still fire."""
    store = Store(tmp_path / "m.sqlite")
    for _ in range(12):
        run_agent(store, evaluate=False)

    profile = build_profile(store.runs("fake_agent"))
    store.save_baseline("fake_agent", 12, profile)
    assert profile["tool_rates"]["lookup"]["empty_rate"] == 0.0

    raised = []
    for _ in range(12):
        middleware = run_agent(store, tool_name="empty_lookup")
        middleware.on_alert = raised.append

    signals = {a["signal"] for a in store.alerts()}
    assert "tool.empty_rate_spike" in signals or "trajectory.unseen_path" in signals


def test_store_is_shared_with_the_stategraph_path(tmp_path):
    """Same schema, same CLI — the adapter is not a parallel universe."""
    store = Store(tmp_path / "m.sqlite")
    run_agent(store)
    runs = store.runs("fake_agent")
    assert set(runs[0]) >= {"run_id", "node_path", "output_keys", "outcome", "tool_events"}
