import requests


def format_prompt(context_chunks, user_query):
    context_blocks = "\n---\n".join([c['chunk'] for c in context_chunks])
    prompt = (
        "You are an expert assistant. Use ONLY the context below to answer.\n"
        "If relevant info is missing, say you don't know.\n"
        f"Context:\n{context_blocks}\n\n"
        f"User question: {user_query}\n\n"
        "Answer:"
    )
    return prompt


def ollama_generate_stream(prompt, model='mistral'):
    # here we use this to connect to local Ollama server, streaming responses
    url = "http://localhost:11434/api/generate"
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": True
    }
    response = requests.post(url, json=payload, stream=True)
    answer = ""
    for line in response.iter_lines():
        if line:
            data = line.decode("utf-8")
            if '"response":"' in data:
                import json
                # sometimes the response comes in multiple lines, so this i used for guard parsing
                try:
                    chunk_text = json.loads(data)["response"]
                    yield chunk_text
                except Exception:
                    continue
