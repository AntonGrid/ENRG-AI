from agent.llm import ask


SYSTEM_PROMPT = """
You are ENRG AI CTO.

Rules:

1. Analyze the project first.
2. Never invent files.
3. Modify existing architecture.
4. Explain your reasoning.
5. Prefer small safe changes.
"""


def plan(context: str, user_request: str):

    prompt = f"""
Project:

{context}

User:

{user_request}
"""

    return ask(prompt, system=SYSTEM_PROMPT)