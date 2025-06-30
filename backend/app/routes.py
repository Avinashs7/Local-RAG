from flask import Blueprint, request, jsonify
from app.embeddings import get_embeddings
from app.retriever import Retriever
from app.generator import generate_answer
from app.utils import load_text_documents
from app.config import Config
import numpy as np

bp = Blueprint("routes", __name__)
retriever = Retriever()

@bp.route("/ingest", methods=["POST"])
def ingest():
    docs = load_text_documents(Config.DOCUMENT_PATH)
    embeddings = get_embeddings(docs)
    retriever.add_documents(embeddings, docs)
    return jsonify({"message": f"{len(docs)} documents ingested."})

@bp.route("/query", methods=["POST"])
def query():
    data = request.json
    question = data.get("question")
    query_embedding = get_embeddings([question])
    results = retriever.retrieve(np.array(query_embedding).astype("float32"))
    context = "\n---\n".join(results)
    answer = generate_answer(context, question)
    return jsonify({"answer": answer})
