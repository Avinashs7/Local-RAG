import requests

def get_embeddings(texts: list[str], model="nomic-embed-text:v1.5") -> list[list[float]]:
    embeddings = []
    for text in texts:
        response = requests.post(
            "http://localhost:11434/api/embeddings",
            json={"model": model, "prompt": text},
            timeout=120
        )
        data = response.json()
        embeddings.append(data["embedding"])
    return embeddings
