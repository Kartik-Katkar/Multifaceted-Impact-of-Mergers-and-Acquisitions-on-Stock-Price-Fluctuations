from flask import Flask, request, render_template, redirect, url_for, flash
import os
from dotenv import load_dotenv
import PyPDF2
import pandas as pd

# for session handling 
import secrets

# for scraping and handling HTML 
import requests
from bs4 import BeautifulSoup
from process_web import scrape_content, extract_text
from process_video import handle_video
from process_image import process_single_image

load_dotenv()

scrape_api_key = os.getenv('SCRAPER_API_KEY')

app = Flask(__name__)
secret_key = secrets.token_bytes(24)
app.secret_key = secret_key
UPLOAD_FOLDER = 'uploads'
TEXT_FILE = 'data.txt'

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route('/')
def index():
    return render_template('upload.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    url = request.form.get('url')
    file = request.files.get('file')

    if url:
        html_content = scrape_content(url,scrape_api_key)
        if html_content:
            text_content = extract_text(html_content)
            append_to_text_file(text_content)
            print(f'Content from {url} has been appended to {TEXT_FILE}.')
        else:
            print(f'Failed to scrape content from {url}.')
    elif file:
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
        # elif file.filename.endswith('.jpeg'):
        elif file.filename.endswith(('.jpg', '.jpeg', '.png', '.bmp', '.gif')):
            process_single_image(file_path, TEXT_FILE)
        
        flash('File successfully processed')
    else:
        flash('No URL or file provided')
    
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
