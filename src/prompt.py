import json
def load(path, n=None):
    rows=[json.loads(l) for l in open(path)]
    return rows[:n] if n else rows
def prompt(text):
    return ("Summarize the following US congressional bill in a clear, factual legislative summary "
            "of at most 200 words. Output only the summary.\n\nBILL:\n"+text)
