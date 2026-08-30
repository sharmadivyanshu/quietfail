"""Graph assembly."""

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph

from . import nodes
from .state import AgentState


def build_graph(checkpointer=None):
    g = StateGraph(AgentState)

    g.add_node("ingest", nodes.ingest)
    g.add_node("extract", nodes.extract)
    g.add_node("validate", nodes.validate)
    g.add_node("classify_exception", nodes.classify_exception)
    g.add_node("resolve", nodes.resolve)
    g.add_node("apply_resolution", nodes.apply_resolution)
    g.add_node("human_review", nodes.human_review)
    g.add_node("post", nodes.post)
    g.add_node("escalate", nodes.escalate)

    g.add_edge(START, "ingest")
    g.add_edge("ingest", "extract")
    g.add_edge("extract", "validate")

    g.add_conditional_edges(
        "validate",
        nodes.route_after_validation,
        # human_review belongs here too: once the resolution round cap is hit,
        # validate routes straight to a human. Omitting it was a latent
        # KeyError that only fired when an invoice failed to converge — found
        # by the failure-injection harness, not by the unit tests.
        {
            "classify_exception": "classify_exception",
            "post": "post",
            "human_review": "human_review",
        },
    )
    g.add_edge("classify_exception", "resolve")
    g.add_conditional_edges(
        "resolve",
        nodes.route_after_resolution,
        {"apply_resolution": "apply_resolution", "human_review": "human_review"},
    )
    # THE CYCLE: fixes flow back through validation. This is what makes it a
    # graph rather than a DAG, and it is the core reason to reach for
    # LangGraph over a LangChain chain.
    g.add_edge("apply_resolution", "validate")
    g.add_conditional_edges(
        "human_review",
        nodes.route_after_human,
        {"post": "post", "escalate": "escalate"},
    )
    g.add_edge("post", END)
    g.add_edge("escalate", END)

    return g.compile(checkpointer=checkpointer or InMemorySaver())


def build_graph_sqlite(db_path: str = "checkpoints.sqlite"):
    """Durable variant. Returns a context manager — the saver owns the conn."""
    from langgraph.checkpoint.sqlite import SqliteSaver

    return SqliteSaver.from_conn_string(db_path)
