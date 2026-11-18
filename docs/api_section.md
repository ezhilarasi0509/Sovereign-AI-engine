
## Try the API

```bash
docker run -p 8080:8080 --env-file .env sovereign-ai-api
curl -s localhost:8080/summarize -H 'Content-Type: application/json' \
  -d '{"text": "<paste a bill here>", "backend": "ollama"}'
```

The response contains `summary`, `backend`, `model`, `latency_s`, `in_tokens` and `out_tokens`. Set `backend` to `ollama`, `vllm` or `gemini` to switch between the local model and the API baseline. Each call is traced in Arize Phoenix.
