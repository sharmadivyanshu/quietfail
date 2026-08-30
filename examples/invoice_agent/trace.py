"""Watch the graph think, one node at a time.

    poetry run python -m invoice_agent.trace unknown-vendor-003

Every LangGraph run is just: a dict of state, passed through functions, where
each function returns a PARTIAL update that gets merged in. This script prints
that merge as it happens, so you can see the state actually change.
"""

import sys

from langgraph.types import Command

from invoice_agent.graph import build_graph

DIM = "\033[2m"
BOLD = "\033[1m"
RESET = "\033[0m"


def fmt(value: object) -> str:
    """Render a state value compactly enough to read in a terminal."""
    if isinstance(value, list):
        if not value:
            return "[]"
        return "\n".join(f"        - {_one(v)}" for v in value)
    return _one(value)


def _one(value: object) -> str:
    if hasattr(value, "model_dump"):
        data = value.model_dump()
        keep = {k: v for k, v in data.items() if v not in (None, [], {})}
        if "rule" in keep:
            return f"{keep['severity']}: {keep['rule']} — {keep['detail']}"
        if "strategy" in keep:
            return f"{keep['strategy']} conf={keep['confidence']} {keep['proposal']}"
        if "invoice_number" in keep:
            return (
                f"{keep['vendor_name']} #{keep['invoice_number']} "
                f"total={keep['total']} vendor_id={keep.get('vendor_id')} "
                f"po={keep.get('po_number')}"
            )
        return str(keep)
    return str(value)


def trace(doc_id: str) -> None:
    graph = build_graph()
    config = {"configurable": {"thread_id": f"trace-{doc_id}"}}

    print(f"\n{BOLD}RUN{RESET} {doc_id}")
    print(f"{DIM}Each block below is ONE node returning ONE partial state update.{RESET}\n")

    step = 0
    # stream_mode="updates" yields {node_name: partial_update} after each node.
    for chunk in graph.stream({"document_uri": doc_id, "tenant_id": "trace"}, config=config):
        for node_name, update in chunk.items():
            step += 1
            if node_name == "__interrupt__":
                payload = update[0].value
                print(f"{BOLD}[{step}] PAUSED{RESET} — graph stopped, state saved to checkpoint")
                print(f"      reason     : {payload['why_escalated']}")
                print(f"      confidence : {payload['confidence']}")
                print("      asking for : a human decision\n")
                continue

            print(f"{BOLD}[{step}] {node_name}{RESET} returned:")
            if not update:
                print(f"    {DIM}(nothing — this node only reads state){RESET}")
            for key, value in (update or {}).items():
                if isinstance(value, list) and value:
                    print(f"    {key} =")
                    print(fmt(value))
                else:
                    print(f"    {key} = {fmt(value)}")
            print()

    state = graph.get_state(config)
    if state.next:
        print(f"{DIM}Graph is parked at {state.next[0]}. Resuming with approval...{RESET}\n")
        for chunk in graph.stream(Command(resume={"approved": True}), config=config):
            for node_name, update in chunk.items():
                step += 1
                print(f"{BOLD}[{step}] {node_name}{RESET} returned: {update}")

    final = graph.get_state(config).values
    print(f"\n{BOLD}FINAL{RESET} outcome={final.get('outcome')} rounds={final.get('iteration')}")
    print(f"{DIM}Full state keys: {sorted(final)}{RESET}\n")


if __name__ == "__main__":
    trace(sys.argv[1] if len(sys.argv) > 1 else "unknown-vendor-003")
