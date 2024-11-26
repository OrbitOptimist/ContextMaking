import requests
from bs4 import BeautifulSoup
from readability import Document
from urllib.parse import urljoin, urlparse
import markdown
import re

class WebpageProcessor:
    def __init__(self):
        self.visited_urls = set()
        self.max_depth = 2
        
    def get_readable_content(self, url, depth=0):
        """Extract readable content from a webpage and its subpages."""
        if depth >= self.max_depth or url in self.visited_urls:
            return ""
            
        try:
            self.visited_urls.add(url)
            response = requests.get(url, headers={'User-Agent': 'Mozilla/5.0'})
            doc = Document(response.text)
            
            # Extract main content
            content = doc.summary()
            title = doc.title()
            
            # Convert to markdown
            soup = BeautifulSoup(content, 'html.parser')
            markdown_content = f"# {title}\n\n"
            
            # Process text and headers
            for element in soup.find_all(['p', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6']):
                if element.name == 'p':
                    markdown_content += f"{element.get_text()}\n\n"
                else:
                    level = int(element.name[1])
                    markdown_content += f"{'#' * level} {element.get_text()}\n\n"
            
            # Find subpages
            if depth < self.max_depth:
                base_domain = urlparse(url).netloc
                links = soup.find_all('a', href=True)
                
                for link in links:
                    href = link['href']
                    full_url = urljoin(url, href)
                    
                    # Only process links from the same domain
                    if urlparse(full_url).netloc == base_domain and full_url not in self.visited_urls:
                        subpage_content = self.get_readable_content(full_url, depth + 1)
                        if subpage_content:
                            markdown_content += f"\n## From subpage: {link.get_text()}\n"
                            markdown_content += subpage_content
            
            return markdown_content
            
        except Exception as e:
            print(f"Error processing {url}: {str(e)}")
            return ""
    
    def reset(self):
        """Reset the visited URLs set."""
        self.visited_urls.clear()
