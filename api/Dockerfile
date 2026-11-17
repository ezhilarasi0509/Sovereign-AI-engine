FROM python:3.12-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY app.py .
ENV OLLAMA_URL=http://host.docker.internal:11434/v1 \
    PHOENIX_ENDPOINT=http://host.docker.internal:6006/v1/traces
EXPOSE 8080
CMD ["uvicorn","app:app","--host","0.0.0.0","--port","8080"]
