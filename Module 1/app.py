from flask import Flask, request, render_template, redirect, url_for, flash
import os
import PyPDF2
import pandas as pd
from bs4 import BeautifulSoup

app = Flask(__name__)
secret_key = os.urandom(24)
app.secret_key = secret_key
UPLOAD_FOLDER = 'uploads'
TEXT_FILE = 'output.txt'

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route('/')
def index():
    return render_template('upload.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        flash('No file part')
        return redirect(request.url)
    
    file = request.files['file']
    
    if file.filename == '':
        flash('No selected file')
        return redirect(request.url)
    
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)

    if os.path.exists(file_path):
        flash('File already exists')
        return redirect(url_for('index'))
    else:
        file.save(file_path)
    
    if file.filename.endswith('.pdf'):
        handle_pdf(file_path)
    elif file.filename.endswith('.csv'):
        handle_csv(file_path)
    elif file.filename.endswith('.mp4'):
        handle_video(file_path)
    elif file.filename.endswith('.html'):
        handle_html(file_path)
    
    flash('File successfully processed')
    return redirect(url_for('index'))

def handle_pdf(file_path):
    reader = PyPDF2.PdfReader(file_path)
    num_pages = len(reader.pages)
    for page_num in range(num_pages):
        page = reader.pages[page_num]
        text = page.extract_text()
        append_to_text_file(text)

def handle_csv(file_path):
    chunksize = 500
    reader = pd.read_csv(file_path, chunksize=chunksize)
    for chunk in reader:
        text = chunk.to_string(index=False)
        append_to_text_file(text)

def handle_video(file_path):
    # Placeholder function for handling video files
    append_to_text_file(f'Video file {file_path} processed.\n')

def handle_html(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        soup = BeautifulSoup(file, 'html.parser')
        text = soup.get_text()
        append_to_text_file(text)

def append_to_text_file(content):
    with open(TEXT_FILE, 'a', encoding='utf-8') as f:
        f.write(content + '\n')

if __name__ == '__main__':
    app.run(debug=True)
