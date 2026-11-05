# ENRG-AI — runtime image.
#
# One image, every entry point the repository has: the self-training loop (the
# default command), the coherence CLI, the forecast CLI, and the suite the host
# runs — so the image can be measured rather than believed. Nothing in it holds
# a key: a venue is configured from the environment at run time, nowhere else.
FROM python:3.12-slim AS runtime

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    ENRG_CACHE_DIR=/data/.cache

WORKDIR /app

# The project and the package it installs. README.md is here because
# `pyproject.toml` names it as the long description: setuptools reads it.
COPY pyproject.toml README.md requirements.txt ./
COPY agent/ ./agent/

# The extras a deploy actually uses: `testnet` is web3.py (a chain), `market`
# is requests (DeFiLlama), `plot` is matplotlib (the figures the CLI can print),
# and `dev` is pytest, so the suite in this image can be run at all. The base
# deps (numpy, httpx, PyNaCl, ollama) come with the project itself.
RUN python -m pip install --upgrade pip setuptools wheel \
 && python -m pip install --no-cache-dir ".[testnet,market,plot,dev]"

# The suite and the CLI scripts are copied in so the image can be *measured*, not
# because a deploy needs them: `pytest -q` inside the container runs every test
# whose dependencies ship in this repository. The ones that need a workspace
# sibling — the `entrain_infer` core, `axis_core`, the scikit-learn datasets —
# skip or fail by name. The whole 692 run on a host that holds the workspace.
COPY tests/ ./tests/
COPY scripts/ ./scripts/

# Non-root, and an owner for the volume that carries the state file.
RUN useradd --create-home --uid 10001 enrg \
 && mkdir -p /data \
 && chown -R enrg:enrg /app /data
USER enrg

# A loop that stopped writing its state file is a loop that is not working;
# the file is written at the end of every cycle, so its age is the heartbeat.
HEALTHCHECK --interval=60s --timeout=10s --start-period=180s --retries=3 \
    CMD ["python", "-c", "import os,sys,time; p='/data/digital_state.json'; sys.exit(0 if os.path.exists(p) and time.time()-os.path.getmtime(p) < 10800 else 1)"]

# Default service: the background self-training loop, one cycle an hour.
# `--offline` is the air-gapped variant (deterministic feeds, no egress).
CMD ["python", "-m", "agent.digital_train.pipeline", \
     "--loop", "--interval", "3600", "--state", "/data/digital_state.json"]
