from flask import Blueprint, request, jsonify
from app.embeddings import get_embeddings
from app.retriever import Retriever
from app.generator import generate_answer
from app.utils import extract_text_from_pdf
from app.config import Config
import numpy as np

bp = Blueprint("routes", __name__)
retriever = Retriever()
@bp.route("/ingest", methods=["POST"])
def ingest():
    file=request.files['file']
    if file:
        file.save(Config.DOCUMENT_PATH)
        pdf_chunks = extract_text_from_pdf(Config.DOCUMENT_PATH)
        pdf_embeddings = get_embeddings(pdf_chunks)
        pdf_embeddings = np.array(pdf_embeddings).astype('float32')
        retriever.add_documents(pdf_embeddings,pdf_chunks)
    else:
        return jsonify({"message":"Please select a pdf file"})
    return jsonify({"message":f"{len(pdf_chunks)} documents ingested"})
        

@bp.route("/query", methods=["POST"])
def query():
    data = request.json
    question = data.get("question")
    query_embedding = get_embeddings([question])
    results = retriever.retrieve(np.array(query_embedding).astype("float32"))
    context = "\n---\n".join(results)
    answer = generate_answer(context, question)
    return jsonify({"answer": answer})
