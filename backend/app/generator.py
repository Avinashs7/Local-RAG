import requests

def generate_answer(context: str, question: str, model="gemma2:2b") -> str:
    prompt = f"Context:\n{context}\n\nQuestion: {question}\nAnswer:"
    response = requests.post(
        "http://localhost:11434/api/generate",
        json={
            "model": model,
            "prompt": prompt,
            "stream": False
        },
        timeout=120
    )
    return response.json()["response"]
