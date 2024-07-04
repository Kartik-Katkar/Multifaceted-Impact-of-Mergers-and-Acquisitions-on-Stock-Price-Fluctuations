from flask import Flask, request, render_template, redirect, url_for, flash
import os
from dotenv import load_dotenv

# for session handling 
import secrets

# for scraping and handling HTML 
import requests
from process_web import scrape_content, extract_text
from process_video import handle_video
from process_image import process_single_image
from handlers import handle_csv,handle_html,handle_pdf,append_to_text_file

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

if __name__ == '__main__':
    app.run(debug=True)
