import os
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import json

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

def process_sec_filings(file_path, chunk_size=1000, output_dir="./FAISS_INDEX",chunk_dir ="./FAISS_INDEX/chunks/"):
    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Read and preprocess text
    text_data = read_text_file(file_path)
    preprocessed_text = preprocess_text(text_data)
    
    # Split preprocessed text into chunks
    chunks = [preprocessed_text[i:i+chunk_size] for i in range(0, len(preprocessed_text), chunk_size)]
    
    # Initialize FAISS index
    index = initialize_faiss_index(384)
    
    # Create a dictionary to store chunk file paths
    chunk_file_map = {}
    
    for idx, chunk in enumerate(chunks):
        # Save each chunk to a separate file
        chunk_file_path = os.path.join(chunk_dir, f"chunk_{idx}.txt")
        with open(chunk_file_path, 'w', encoding='utf-8') as chunk_file:
            chunk_file.write(chunk)
        
        # Add the chunk to the FAISS index
        embeddings = model.encode([chunk])
        embeddings_np = np.array(embeddings)
        index.add(embeddings_np)
        
        # Map the index to the chunk file path
        chunk_file_map[idx] = chunk_file_path
    
    # Save the FAISS index
    faiss.write_index(index, os.path.join(output_dir, "faiss_index.index"))
    
    # Save the chunk file map to a JSON file
    with open(os.path.join(output_dir, "chunk_file_map.json"), 'w', encoding='utf-8') as map_file:
        json.dump(chunk_file_map, map_file)

if __name__ == "__main__":
    sec_file_path = './Test Data/data.txt'
    process_sec_filings(sec_file_path)