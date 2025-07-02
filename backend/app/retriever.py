import faiss
import os
import numpy as np
from app.config import Config

class Retriever:
    def __init__(self):
        self.index = None
        self.documents = []
        self.load_index()

    def load_index(self):
        index_path = Config.INDEX_PATH
        if os.path.exists(index_path):
            self.index = faiss.read_index(index_path)
            with open(index_path.replace(".faiss", ".meta"), "r", encoding="utf-8") as f:
                self.documents = [line.strip() for line in f.readlines()]
        else:
            self.index = faiss.IndexFlatL2(768)  

    def save_index(self):
        abs_path = os.path.abspath(Config.INDEX_PATH)
        index_dir = os.path.dirname(abs_path)

        # Ensure the parent directory exists
        if not os.path.exists(index_dir):
            os.makedirs(index_dir, exist_ok=True)
            print(f"[Retriever] Created directory: {index_dir}")

        print(f"[Retriever] Saving Faiss index at: {abs_path}")
        faiss.write_index(self.index, abs_path)

    def add_documents(self, embeddings: np.ndarray, texts: list[str]):
        self.index.add(embeddings)
        self.documents.extend(texts)
        self.save_index()

    def retrieve(self, query_embedding: np.ndarray, k: int = 5):
        distances, indices = self.index.search(query_embedding, k)
        return [self.documents[i] for i in indices[0]]
