# AIMLP platform — Dockerfile
# Multi-stage: build the React frontend, then bundle it into the Python
# backend image so a single container serves both /app (React) and the
# API. Buildable/correct config; the FastAPI half was verified running
# directly via uvicorn in the sandbox this was developed in (no Docker
# daemon available there) -- see README for exactly what was and wasn't
# tested through Docker itself.

FROM node:20-slim AS frontend-build
WORKDIR /frontend
COPY frontend/package*.json ./
RUN npm install
COPY frontend/ ./
RUN npm run build

FROM python:3.11-slim
WORKDIR /app

# libgomp1 is required by lightgbm/xgboost/catboost at runtime
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .
COPY --from=frontend-build /frontend/dist ./frontend/dist

EXPOSE 8000
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
