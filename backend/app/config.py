import os

class Config:
    OLLAMA_EMBED_URL = "http://localhost:11434/api/embeddings"
    OLLAMA_GEN_URL = "http://localhost:11434/api/generate"
    INDEX_PATH=os.path.join("vectorstore", "faiss_index", "index.faiss")
    DOCUMENT_PATH=os.path.join("data", "documents")
