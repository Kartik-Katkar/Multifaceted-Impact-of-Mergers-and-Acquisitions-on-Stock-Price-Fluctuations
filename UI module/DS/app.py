import os
from flask import Flask, render_template, request
from sentence_transformers import SentenceTransformer
import faiss
import json
import numpy as np
from RAG_prompt import make_rag_prompt,generate_answer

app = Flask(__name__)
model = SentenceTransformer('all-MiniLM-L6-v2')

# Load FAISS index
faiss_index_path = './FAISS_INDEX/faiss_index.index'
faiss_index = faiss.read_index(faiss_index_path)

chunk_file_map_path = 'FAISS_INDEX/chunk_file_map.json'

with open(chunk_file_map_path, 'r', encoding='utf-8') as map_file:
    chunk_file_map = json.load(map_file)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/ask', methods=['POST'])
def ask():
    if request.method == 'POST':
        question = request.form['question']

        # Encode user query
        query_embedding = model.encode(question)

        # Perform FAISS search
        k = 5  # Number of top results to retrieve
        distances, indices = faiss_index.search(np.array([query_embedding]), k)

        # Retrieve and format results
        results = []
        for idx in indices[0]:
            chunk_file_path = chunk_file_map[str(idx)]
            with open(chunk_file_path, 'r', encoding='utf-8') as chunk_file:
                chunk_text = chunk_file.read()
                results.append(chunk_text)
        
        final_prompt = make_rag_prompt(question,results)
        answer = generate_answer(final_prompt)

        return render_template('result.html', question=question, result_text=answer)

if __name__ == '__main__':
    app.run(debug=True, port=5002)