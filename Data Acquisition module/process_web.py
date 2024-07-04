import requests
from bs4 import BeautifulSoup

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