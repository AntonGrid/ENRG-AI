"""Allow ``python -m agent.trade`` (closed economic loop)."""

from agent.trade.loop import main

if __name__ == "__main__":
    raise SystemExit(main())
