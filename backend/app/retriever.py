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
        faiss.write_index(self.index, Config.INDEX_PATH)
        with open(Config.INDEX_PATH.replace(".faiss", ".meta"), "w", encoding="utf-8") as f:
            for doc in self.documents:
                f.write(f"{doc}\n")

    def add_documents(self, embeddings: np.ndarray, texts: list[str]):
        self.index.add(embeddings)
        self.documents.extend(texts)
        self.save_index()

    def retrieve(self, query_embedding: np.ndarray, k: int = 5):
        distances, indices = self.index.search(query_embedding, k)
        return [self.documents[i] for i in indices[0]]
