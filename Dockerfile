# ── Stage: runtime ────────────────────────────────────────────────────────────
FROM python:3.10-slim

LABEL org.opencontainers.image.title="server-thermal-sysadmin"
LABEL org.opencontainers.image.description="Meta PyTorch OpenEnv Hackathon — Automated IT Sysadmin Workspace"
LABEL org.opencontainers.image.version="1.0.0"

# ── System dependencies ───────────────────────────────────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
        ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# ── Working directory ─────────────────────────────────────────────────────────
WORKDIR /app

# ── Python dependencies ───────────────────────────────────────────────────────
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# ── Application source ────────────────────────────────────────────────────────
COPY openenv.yaml  ./
COPY laptop_env.py ./
COPY server.py     ./
COPY inference.py  ./
COPY baseline.py   ./
COPY demo.py       ./
COPY README.md     ./

# ── Runtime environment ───────────────────────────────────────────────────────
ENV PYTHONUNBUFFERED=1
ENV PYTHONIOENCODING=utf-8

# HuggingFace Spaces uses port 7860
EXPOSE 7860

# ── Health-check: POST /reset with empty body must return 200 ─────────────────
HEALTHCHECK --interval=15s --timeout=10s --start-period=10s --retries=5 \
    CMD python -c "\
import urllib.request, json; \
req = urllib.request.Request('http://localhost:7860/reset', \
  data=json.dumps({}).encode(), \
  headers={'Content-Type': 'application/json'}, method='POST'); \
assert urllib.request.urlopen(req).status == 200" \
    || exit 1

# ── Entrypoint — run the FastAPI environment server ───────────────────────────
CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "7860", "--log-level", "info"]
