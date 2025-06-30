import os

def load_text_documents(doc_dir):
    texts = []
    for filename in os.listdir(doc_dir):
        with open(os.path.join(doc_dir, filename), "r", encoding="utf-8") as f:
            texts.append(f.read())
    return texts
