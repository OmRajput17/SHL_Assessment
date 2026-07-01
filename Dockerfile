# ── Build stage ──────────────────────────────────────────────────────────────
FROM python:3.11-slim AS builder

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ── Runtime stage ─────────────────────────────────────────────────────────────
FROM python:3.11-slim

WORKDIR /app

# Copy installed packages from builder
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Copy application code
COPY . .

# HuggingFace Spaces uses 7860; Render/Railway inject $PORT at runtime.
# Default to 7860 so the image works on HF Spaces without modification.
ENV PORT=7860

EXPOSE 7860

# GROQ_API_KEY must be supplied at runtime (env var or platform secret).
# Never bake it into the image.

CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT}"]
