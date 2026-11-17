import os, time
from dotenv import load_dotenv; load_dotenv()
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from openai import OpenAI
from phoenix.otel import register
register(project_name="sovereign-ai",
         endpoint=os.getenv("PHOENIX_ENDPOINT","http://localhost:6006/v1/traces"),
         auto_instrument=True)
BACKENDS={
 "ollama":dict(base_url=os.getenv("OLLAMA_URL","http://localhost:11434/v1"),key="ollama",
               model=os.getenv("OLLAMA_MODEL","billsum-q4b")),
 "vllm":dict(base_url=os.getenv("VLLM_URL","http://localhost:8000/v1"),key="x",model="m"),
 "gemini":dict(base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
               key=os.getenv("GEMINI_API_KEY","x"),model=os.getenv("GEMINI_MODEL","gemini-3.1-flash-lite")),
}
PROMPT=("Summarize the following US congressional bill in a clear, factual legislative summary "
        "of at most 200 words. Output only the summary.\n\nBILL:\n")
app=FastAPI(title="Sovereign AI summarizer")
class Req(BaseModel):
    text: str
    backend: str = os.getenv("DEFAULT_BACKEND","ollama")
@app.get("/health")
def health(): return {"ok":True,"backends":list(BACKENDS)}
@app.post("/summarize")
def summarize(r: Req):
    b=BACKENDS.get(r.backend)
    if not b: raise HTTPException(400,f"backend must be one of {list(BACKENDS)}")
    c=OpenAI(base_url=b["base_url"],api_key=b["key"],timeout=300)
    t=time.time()
    x=c.chat.completions.create(model=b["model"],temperature=0,max_tokens=300,
        messages=[{"role":"user","content":PROMPT+r.text}])
    return {"summary":x.choices[0].message.content,"backend":r.backend,"model":b["model"],
            "latency_s":round(time.time()-t,2),
            "in_tokens":x.usage.prompt_tokens,"out_tokens":x.usage.completion_tokens}
