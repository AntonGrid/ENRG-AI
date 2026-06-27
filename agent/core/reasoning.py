from agent.core.intent import detect
from agent.context.builder import build_context
from agent.graph.manager import graph
from agent.graph.query import GraphQuery
from agent.index.search import search


def reason(query: str):

    task = detect(query)

    matches = search(task["target"])

    plan = {
        "intent": task["intent"],
        "query": task["target"],
        "matches": matches,
        "context": [],
        "graph": None,
        "symbol": None,
        "file": None,
    }

    if matches:

        match = matches[0]

        plan["symbol"] = match.get("matched_symbol")
        plan["file"] = match.get("path")

        if plan["symbol"]:

            g = graph()

            plan["graph"] = GraphQuery(g).impact(
                plan["symbol"]
            )

    plan["context"] = build_context(plan)

    return plan
