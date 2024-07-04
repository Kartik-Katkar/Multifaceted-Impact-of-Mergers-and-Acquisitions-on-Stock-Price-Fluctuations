import requests
from bs4 import BeautifulSoup
import threading
import time
from handlers import append_to_text_file

# Global variables for threading and stopping mechanism
scraping_thread = None
stop_scraping = threading.Event()

# Function to scrape content from a URL using ScraperAPI
def scrape_content(url, api_key):
    scraperapi_url = f'http://api.scraperapi.com?api_key={api_key}&url={url}'
    response = requests.get(scraperapi_url)
    if response.status_code == 200:
        return response.text
    else:
        return None

# Function to extract text content from HTML
def extract_text(html_content):
    soup = BeautifulSoup(html_content, 'html.parser')
    return soup.get_text(separator='\n')

# Function to scrape periodically
def scrape_periodically(url, api_key):
    global scraping_thread, stop_scraping
    while not stop_scraping.is_set():
        html_content = scrape_content(url, api_key)
        if html_content:
            text_content = extract_text(html_content)
            append_to_text_file(text_content)
            print(f'Content from {url} has been appended.')
        else:
            print(f'Failed to scrape content from {url}.')
        time.sleep(10)  # Sleep for 5 minutes (300 seconds)

# Function to start scraping process
def start_scraping(url, api_key):
    global scraping_thread
    if scraping_thread and scraping_thread.is_alive():
        print('Scraping is already running.')
    else:
        scraping_thread = threading.Thread(target=scrape_periodically, args=(url, api_key))
        scraping_thread.start()
        print(f'Started scraping {url} every 10 seconds')

# Function to stop scraping process
def stop_scraping_process():
    global stop_scraping, scraping_thread
    stop_scraping.set()
    if scraping_thread and scraping_thread.is_alive():
        scraping_thread.join()
        print('Scraping stopped.')