"""Run every fixture through the graph and show what happened.

poetry run python -m invoice_agent
"""

from langgraph.types import Command

from invoice_agent.graph import build_graph

FIXTURES = [
    "clean-001",
    "missing-gl-002",
    "unknown-vendor-003",
    "totals-broken-004",
    "duplicate-005",
]


def show(doc_id: str, graph) -> None:
    config = {"configurable": {"thread_id": doc_id}}
    result = graph.invoke({"document_uri": doc_id, "tenant_id": "demo"}, config=config)
    state = graph.get_state(config)

    print(f"\n{'=' * 70}\n{doc_id}\n{'-' * 70}")

    errors = [f for f in result.get("findings", []) if f.severity == "error"]
    for finding in errors:
        print(f"  [error] {finding.rule}: {finding.detail}")
    if not errors:
        print("  no validation errors")

    if result.get("exception_class"):
        print(f"  class      : {result['exception_class']}")
    for res in result.get("resolutions", []):
        print(f"  attempt    : {res.strategy} (confidence {res.confidence}) -> {res.proposal}")
    print(f"  rounds     : {result.get('iteration', 0)}")

    if state.next:
        payload = state.tasks[0].interrupts[0].value
        print(f"  --> PAUSED at {state.next[0]}: {payload['why_escalated']}")
        result = graph.invoke(Command(resume={"approved": True, "reviewer": "demo"}), config=config)
        print(f"      resumed -> {result['outcome']}")
    else:
        print(f"  outcome    : {result.get('outcome')}")


def main() -> None:
    graph = build_graph()
    for doc_id in FIXTURES:
        show(doc_id, graph)
    print(f"\n{'=' * 70}")


if __name__ == "__main__":
    main()
