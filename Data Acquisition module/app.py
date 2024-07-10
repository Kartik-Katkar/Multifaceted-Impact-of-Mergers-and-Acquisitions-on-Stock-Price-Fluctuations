from flask import Flask, request, render_template, redirect, url_for, flash
import os
from dotenv import load_dotenv

# for session handling 
import secrets

# for scraping and handling HTML 
import requests
import threading
from process_web import scrape_content, extract_text, start_scraping, stop_scraping_process
from process_video import handle_video
from process_image import process_single_image
from handlers import handle_csv,handle_html,handle_pdf
from Ragfilter import classify_and_append

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

@app.route('/stop_scraping', methods=['POST'])
def stop_scraping():
    stop_scraping_process()
    flash('Scraping stopped.')
    return redirect(url_for('index'))

@app.route('/upload', methods=['POST'])
def upload_file():

    global scraping_thread
    url = request.form.get('url')
    file = request.files.get('file')

    if url:
        if not url.startswith('http'):
            url = 'http://' + url
        start_scraping(url, scrape_api_key)
        flash(f'Started scraping {url} every 5 minutes.')

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
