import os

class Config:
    OLLAMA_EMBED_URL = "http://localhost:11434/api/embeddings"
    OLLAMA_GEN_URL = "http://localhost:11434/api/generate"
    BASE_DIR = os.path.abspath(os.path.dirname(__file__))
    INDEX_PATH = os.path.join(BASE_DIR, "..", "vectorstore", "faiss_index", "index.faiss")
    DOCUMENT_PATH=os.path.join("data","documents","uploaded.pdf")
