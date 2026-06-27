COMMANDS = {
    "analyze": "analyze",
    "impact": "impact",
    "find": "find",
    "explain": "explain",
    "fix": "fix",
}


def detect(query: str):

    query = query.strip()

    if not query:
        return {
            "intent": "analyze",
            "target": "",
        }

    parts = query.split(maxsplit=1)

    if len(parts) == 1:
        return {
            "intent": "analyze",
            "target": parts[0],
        }

    cmd = parts[0].lower()

    if cmd in COMMANDS:

        return {
            "intent": COMMANDS[cmd],
            "target": parts[1],
        }

    return {
        "intent": "analyze",
        "target": query,
    }
