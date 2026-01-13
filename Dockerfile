# ===== MCU Purple Agent Dockerfile =====
FROM python:3.10-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# system deps
RUN apt-get update && apt-get install -y \
    git curl ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# deps
COPY requirements.txt .
RUN pip install --upgrade pip \
 && pip install --no-cache-dir -r requirements.txt

# app
COPY src ./src

EXPOSE 9019

CMD ["python", "-m", "src.server.app", "--host", "0.0.0.0", "--port", "9019"]
