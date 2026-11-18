cd ~/sovereign-ai; source .venv/bin/activate
while true; do
  if python -W ignore - <<'PY' 2>/dev/null
from dotenv import load_dotenv; load_dotenv(".env")
from google import genai
c=genai.Client(); c.models.generate_content(model="gemini-3.1-flash-lite",contents="Say hi"); print("ok")
PY
  then break; fi
  echo "quota not back $(date)"; sleep 600
done
python -W ignore -u src/run_gemini_fewshot.py
bash src/integrate_2shot.sh
