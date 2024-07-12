import os
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

load_dotenv()

model = SentenceTransformer('all-MiniLM-L6-v2')

def read_text_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        text_data = file.read()
    return text_data

def preprocess_text(text):
    tokens = word_tokenize(text)
    tokens = [token.lower() for token in tokens if token.isalnum()]
    stop_words = set(stopwords.words('english'))
    tokens = [token for token in tokens if token not in stop_words]
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(token) for token in tokens]
    preprocessed_text = ' '.join(tokens)
    return preprocessed_text

def initialize_faiss_index(embedding_dim):
    index = faiss.IndexFlatL2(embedding_dim)
    return index

def process_sec_filings(file_path, chunk_size=10000):
    text_data = read_text_file(file_path)
    preprocessed_text = preprocess_text(text_data)
    chunks = [preprocessed_text[i:i+chunk_size] for i in range(0, len(preprocessed_text), chunk_size)]
    index = initialize_faiss_index(384)
    for chunk in chunks:
        embeddings = model.encode([chunk])
        embeddings_np = np.array(embeddings)
        index.add(embeddings_np)
    faiss.write_index(index, "./FAISS_INDEX/faiss_index.index")

if __name__ == "__main__":
    sec_file_path = './Test Data/data.txt'
    process_sec_filings(sec_file_path)
