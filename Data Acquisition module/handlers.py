import PyPDF2
from bs4 import BeautifulSoup
import pandas as pd
from Ragfilter import classify_and_append

TEXT_FILE = 'data.txt'

def handle_pdf(file_path):
    reader = PyPDF2.PdfReader(file_path)
    num_pages = len(reader.pages)
    for page_num in range(num_pages):
        page = reader.pages[page_num]
        text = page.extract_text()
        classify_and_append(text)

def handle_csv(file_path):
    chunksize = 500
    reader = pd.read_csv(file_path, chunksize=chunksize)
    for chunk in reader:
        text = chunk.to_string(index=False)
        classify_and_append(text)

def handle_html(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        soup = BeautifulSoup(file, 'html.parser')
        text = soup.get_text()
        classify_and_append(text)

# def append_to_text_file(content):
#     with open(TEXT_FILE, 'a', encoding='utf-8') as f:
#         f.write(content + '\n')